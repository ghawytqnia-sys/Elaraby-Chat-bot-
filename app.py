import asyncio
import json
import re
import time
import uuid
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Elaraby AI Enterprise Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static and templates

templates = Jinja2Templates(directory="templates")

# ==========================================
# CONFIGURATION
# ==========================================
FIREBASE_URL = "https://elaraby-products-default-rtdb.firebaseio.com/products.json"
GEMINI_API_KEY = "AIzaSyBizWjfHN6t84gpHHEH2SFhF079bS-YwkU"
GEMINI_MODELS_FALLBACK = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest", 
    "gemini-pro"
]

# ==========================================
# CACHE & MEMORY
# ==========================================
GLOBAL_STATE = {
    "cached_products": [],
    "search_cache": {},
    "last_fetch": 0,
}
PRODUCT_CACHE_TTL = 86400  # 24 hours
SESSIONS = {}

# ==========================================
# MODELS
# ==========================================
class ChatRequest(BaseModel):
    text: str
    session_id: str
    pending_action: Optional[Dict[str, Any]] = None

class SessionMemory:
    def __init__(self):
        self.short_term_history = []
        self.long_term_facts = []
        self.session_summary = ""
        self.last_product_list = []
        
    def push(self, role: str, content: str):
        self.short_term_history.append({"role": role, "content": content, "ts": time.time()})
        if len(self.short_term_history) > 40:
            self.summarize_old_messages()
            
    def summarize_old_messages(self):
        keep_count = 30
        to_summarize = self.short_term_history[:-keep_count]
        self.short_term_history = self.short_term_history[-keep_count:]
        user_msgs = [m["content"][:80] for m in to_summarize if m["role"] == "user"][-8:]
        if user_msgs:
            lines = "\n".join([f"• {m}" for m in user_msgs])
            self.session_summary = f"ملخص المحادثة السابقة:\n{lines}"
            
    def extract_facts(self, text: str):
        patterns = [
            (r'ميزانيت[ي]?\s*([\d,]+)', "الميزانية"),
            (r'(سامسونج|اللجي|توشيبا|شارب|كريير|يونيون|جنرال|هايسنس|ميديا)', "العلامة المفضلة"),
            (r'(ثلاج[ه]?|غسال[ه]?|شاش[ه]?|بوتاجاز|مكيف|سخان)', "الفئة"),
            (r'اللون\s+(الأبيض|الأسود|الفضي|الرمادي|الذهبي|الأخضر)', "اللون المفضل"),
        ]
        for p, label in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                fact = f"{label}: {match.group(1)}"
                if fact not in self.long_term_facts:
                    if len(self.long_term_facts) >= 30:
                        self.long_term_facts.pop(0)
                    self.long_term_facts.append(fact)

# ==========================================
# PRODUCT SEARCH ENGINE
# ==========================================
class ProductSearch:
    STOP_WORDS = {"او","مع","في","من","عن","على","قارن","بين","ماهو","افضل","سعر","مواصفات","هل","ما","هذا","اريد","ممكن","عاوز","هات","ورينى","كام","price","specs","show","me","the","a","an","is","to","of","for","with","and","or"}
    SYNONYMS = {
        "تليفون":"موبايل", "فون":"موبايل", "جوال":"موبايل", "هاتف":"موبايل", "smartphone":"موبايل", "phone":"موبايل", "mobile":"موبايل",
        "كمبيوتر":"لاب", "laptop":"لاب", "notebook":"لاب", "pc":"لاب", "نوت بوك":"لاب",
        "تكييف":"مكيف", "ايركوندشن":"مكيف", "ac":"مكيف", "سبليت":"مكيف",
        "ثلاجه":"ثلاجة", "فريزر":"ثلاجة", "fridge":"ثلاجة", "refrigerator":"ثلاجة",
        "غساله":"غسالة", "washer":"غسالة", "washing":"غسالة",
        "شاشه":"شاشة", "تلفزيون":"شاشة", "تليفزيون":"شاشة", "tv":"شاشة", "television":"شاشة",
        "رخيص":"اقتصادي", "ارخص":"اقتصادي", "cheap":"اقتصادي", "budget":"اقتصادي",
        "غالي":"بريميوم", "الاغلى":"بريميوم", "premium":"بريميوم", "luxury":"بريميوم",
        "جيمنج":"العاب", "gaming":"العاب", "game":"العاب",
        "تصوير":"كاميرا", "صورة":"كاميرا", "photo":"كاميرا", "camera":"كاميرا",
        "بطاريه":"بطارية", "battery":"بطارية",
        "تصميم":"جرافيك", "design":"جرافيك", "graphics":"جرافيك"
    }
    
    @staticmethod
    def normalize(text: str) -> str:
        text = re.sub(r'[أإآ]', 'ا', str(text))
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'ى', 'ي', text)
        return re.sub(r'\s+', ' ', text).strip().lower()

    @staticmethod
    def expand_synonyms(text: str) -> str:
        norm = ProductSearch.normalize(text)
        for k, v in ProductSearch.SYNONYMS.items():
            norm = re.sub(rf'\b{k}\b', f"{k} {v}", norm)
        return norm

    @staticmethod
    def search(query: str, top_k: int = 3, session: SessionMemory = None) -> List[Dict]:
        cache_key = f"{query}|{top_k}"
        if cache_key in GLOBAL_STATE["search_cache"]:
            return GLOBAL_STATE["search_cache"][cache_key]

        expanded = ProductSearch.expand_synonyms(query)
        norm_q = ProductSearch.normalize(query)
        keywords = [w for w in expanded.split() if len(w) > 1 and w not in ProductSearch.STOP_WORDS]
        
        if not keywords: return []

        scored = []
        for p in GLOBAL_STATE["cached_products"]:
            n_name = ProductSearch.normalize(p.get("name", ""))
            n_sku = ProductSearch.normalize(p.get("sku", ""))
            n_brand = ProductSearch.normalize(p.get("brand", ""))
            n_specs = ProductSearch.normalize(p.get("specs", ""))[:500]
            
            score = 0
            if n_sku and n_sku in keywords: score += 50
            
            for kw in keywords:
                if n_name == norm_q: score += 25
                elif n_name.startswith(norm_q): score += 18
                elif kw in n_name: score += 10
                if kw in n_sku: score += 12
                if kw in n_brand: score += 6
                if kw in n_specs: score += 3
            
            if score > 0:
                p_copy = dict(p)
                p_copy["_score"] = score
                scored.append(p_copy)
                
        scored.sort(key=lambda x: x["_score"], reverse=True)
        results = scored[:top_k]
        GLOBAL_STATE["search_cache"][cache_key] = results
        if session and results:
            session.last_product_list = results
        return results

    @staticmethod
    def build_context(products: List[Dict]) -> str:
        if not products:
            return "لا يوجد منتج مطابق في قاعدة البيانات."
        res = []
        for i, p in enumerate(products):
            stock = p.get('stock')
            stock_str = str(stock) if stock is not None else "غير محدد"
            res.append(f"=== المنتج {i+1} ===\nالاسم: {p.get('name')}\nالماركة: {p.get('brand')}\nالسعر: {p.get('price')}\nالكود: {p.get('sku')}\nالضمان: {p.get('warranty')}\nالمخزون: {stock_str}\nالمواصفات:\n{p.get('specs')}")
        return "\n\n".join(res)

# ==========================================
# INTENT DETECTOR & LOCAL ENGINE
# ==========================================
class IntentDetector:
    INTENTS = {
        "PRICE": {"kw": ["سعر","بكام","كام","تمن","ثمن","بكم","price"]},
        "SPECS": {"kw": ["مواصفات","مواصفه","تفاصيل","خصائص","المواصفات","specs","details"]},
        "IMAGES": {"kw": ["صور","صورة","الصور","ورينى","ورني","اعرض","images","pic","show"]},
        "COMPARE": {"kw": ["قارن","مقارنة","الفرق","أفضل من","ولا","بين","compare","vs"]},
        "RECOMMEND": {"kw": ["بديل","اقتراح","انصحني","افضل","أفضل","ايه الافضل","recommend"]},
        "AVAILABILITY": {"kw": ["متوفر","موجود","المخزون","عندكم","هل في","stock","available"]},
        "WARRANTY": {"kw": ["ضمان","الضمان","كفالة","صيانة","الكفالة","warranty"]},
        "PERSUADE": {"kw": ["اقنع","أقنعني","مميزات","ليه اشتري","persuade"]},
    }

    @staticmethod
    def detect(text: str) -> str:
        norm = ProductSearch.normalize(text)
        best_type = "GENERAL"
        max_score = 0
        for type_, data in IntentDetector.INTENTS.items():
            score = sum(1 for k in data["kw"] if k in norm)
            if score > max_score:
                max_score = score
                best_type = type_
        return best_type

class LocalEngine:
    @staticmethod
    def generate(intent: str, query: str, session: SessionMemory) -> Optional[Dict]:
        results = ProductSearch.search(query, 1, session)
        if not results: return None
        
        p = results[0]
        html = ""
        if intent == "PRICE":
            html = f"سعر **{p.get('name')}** حالياً هو:\n<span class=\"price-tag\"><i class=\"fas fa-tag\"></i> السعر: {p.get('price')}</span>"
        elif intent == "SPECS":
            html = f"**أهم مواصفات {p.get('name')}:**\n\n{p.get('specs')}\n\n*الكود المرجعي:* `{p.get('sku')}`"
        elif intent == "AVAILABILITY":
            stock = p.get('stock')
            if stock is not None:
                html = f"**حالة المخزون لـ {p.get('name')}:**\n✅ المنتج متوفر حالياً. (الكمية: {stock})" if int(stock) > 0 else f"**حالة المخزون لـ {p.get('name')}:**\n❌ المنتج غير متوفر."
            else:
                html = f"**حالة المخزون لـ {p.get('name')}:**\n📦 يرجى مراجعة أقرب فرع لمعرفة التوفر."
        elif intent == "WARRANTY":
            html = f"**الضمان الخاص بـ {p.get('name')}:**\n🛡️ {p.get('warranty', 'يخضع لضمان العربي الافتراضي.')}"
        elif intent == "IMAGES":
            return {"type": "gallery", "product": p}
            
        if html:
            return {"type": "local", "html": html}
        return None

# ==========================================
# COMPARISON ENGINE
# ==========================================
class ComparisonEngine:
    @staticmethod
    def format_specs(specs: str) -> str:
        if not specs: return '-'
        lines = [re.sub(r'^[-•*]', '', l).strip() for l in specs.split('\n') if len(l) > 2]
        if len(lines) < 2:
            lines = [l.strip() for l in re.split(r'[,،]', specs) if len(l) > 2]
        if not lines: return '-'
        
        visible = lines[:6]
        html = f'<ul style="padding-right:18px; margin:0; text-align:right; list-style-type:disc;" title="{specs[:100]}...">'
        for l in visible:
            html += f'<li style="margin-bottom:5px; font-size:12.5px; line-height:1.4;">{l}</li>'
        html += '</ul>'
        if len(lines) > 6:
            html += f'<div style="color:var(--brand-1);font-size:10.5px;margin-top:6px;font-weight:bold;">+ {len(lines)-6} مواصفات أخرى</div>'
        return html

    @staticmethod
    def generate_local_table(prodA_name: str, prodB_name: str, session: SessionMemory) -> str:
        rA = ProductSearch.search(prodA_name, 1, session)
        rB = ProductSearch.search(prodB_name, 1, session)
        if not rA or not rB:
            return '<div class="error-notice"><i class="fas fa-exclamation-triangle"></i> عذراً، لم أتمكن من العثور على أحد المنتجين لتنفيذ المقارنة.</div>'
        
        pA, pB = rA[0], rB[0]
        nA = int(re.sub(r'\D', '', str(pA.get('price'))) or 0)
        nB = int(re.sub(r'\D', '', str(pB.get('price'))) or 0)
        
        priceA = f"<span style='font-weight:bold'>{pA.get('price')}</span>"
        priceB = f"<span style='font-weight:bold'>{pB.get('price')}</span>"
        if nA > 0 and nB > 0:
            if nA < nB: priceA = f"<span style='color:#06D6A0;font-weight:bold'>{pA.get('price')} <i class='fas fa-arrow-down' title='أرخص'></i></span>"
            elif nB < nA: priceB = f"<span style='color:#06D6A0;font-weight:bold'>{pB.get('price')} <i class='fas fa-arrow-down' title='أرخص'></i></span>"
            
        html = f"""
        <strong>مقارنة المواصفات الأساسية 📊</strong><br>
        <div class="ai-compare-table-wrapper">
            <table class="ai-compare-table" dir="rtl">
                <thead><tr><th>الميزة</th><th>{pA.get('name')}</th><th>{pB.get('name')}</th></tr></thead>
                <tbody>
                    <tr><td>السعر</td><td>{priceA}</td><td>{priceB}</td></tr>
                    <tr><td>الماركة</td><td>{pA.get('brand','-')}</td><td>{pB.get('brand','-')}</td></tr>
                    <tr><td>الضمان</td><td>{pA.get('warranty','-')}</td><td>{pB.get('warranty','-')}</td></tr>
                    <tr><td>المواصفات</td><td>{ComparisonEngine.format_specs(pA.get('specs'))}</td><td>{ComparisonEngine.format_specs(pB.get('specs'))}</td></tr>
                </tbody>
            </table>
        </div>
        <p style="margin-top:8px;font-size:11.5px;color:var(--text-sec);font-weight:bold;">* تمت المقارنة محلياً بسرعة.</p>
        """
        return html

# ==========================================
# GEMINI API INTEGRATION
# ==========================================
async def stream_gemini(system_prompt: str, user_prompt: str, session: SessionMemory, ignore_history: bool, temperature: float = 0.3):
    contents = []
    
    full_prompt = f"[تعليمات برمجية لك كذكاء اصطناعي]\n{system_prompt}\n\n[سؤال العميل المطلوب الإجابة عليه]\n{user_prompt}"
    
    if not ignore_history:
        history = session.short_term_history[-12:]
        last_role = None
        for msg in history:
            mapped_role = "model" if msg["role"] == "assistant" else "user"
            if mapped_role == last_role and contents:
                contents[-1]["parts"][0]["text"] += f"\n\n{msg['content']}"
            else:
                contents.append({"role": mapped_role, "parts": [{"text": msg["content"]}]})
                last_role = mapped_role
                
        # cleanup roles
        while contents and contents[0]["role"] != "user": contents.pop(0)
        if contents and contents[-1]["role"] != "user": contents.pop()
        
    contents.append({"role": "user", "parts": [{"text": full_prompt}]})
    
    payload = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 1500}
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        last_error = ""
        for model in GEMINI_MODELS_FALLBACK:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
            try:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        last_error = await response.aread()
                        continue
                    
                    full_text = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if not data_str or data_str == "[DONE]": continue
                            try:
                                data = json.loads(data_str)
                                text_part = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                                if text_part:
                                    full_text += text_part
                                    yield f"data: {json.dumps({'type': 'token', 'text': text_part})}\n\n"
                            except Exception: pass
                            
                    session.push("assistant", full_text)
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
            except Exception as e:
                last_error = str(e)
                
        yield f"data: {json.dumps({'type': 'error', 'message': f'فشل الاتصال بالذكاء الاصطناعي: {last_error}'})}\n\n"

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/init")
async def init_data():
    now = time.time()
    if not GLOBAL_STATE["cached_products"] or (now - GLOBAL_STATE["last_fetch"] > PRODUCT_CACHE_TTL):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(FIREBASE_URL)
                data = resp.json()
                if data:
                    products = []
                    for k, p in data.items():
                        name = p.get("name", p.get("title", p.get("productName", p.get("اسم", ""))))
                        if not name: continue
                        products.append({
                            "id": k,
                            "name": name,
                            "price": p.get("price", p.get("Price", p.get("سعر", "غير متوفر"))),
                            "specs": re.sub(r'<[^>]*>', '', str(p.get("details", p.get("specs", p.get("desc", ""))))).strip(),
                            "sku": p.get("id", p.get("sku", p.get("SKU", p.get("code", k)))),
                            "brand": p.get("brand", p.get("Brand", p.get("ماركة", ""))),
                            "stock": p.get("stock", p.get("quantity", None)),
                            "warranty": p.get("warranty", p.get("الضمان", "")),
                            "images": p.get("images", p.get("imgs", p.get("صور", []))),
                            "colors": p.get("colors", p.get("ألوان", [])),
                        })
                    GLOBAL_STATE["cached_products"] = products
                    GLOBAL_STATE["last_fetch"] = now
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)})
            
    return {"status": "success", "count": len(GLOBAL_STATE["cached_products"])}

@app.get("/api/autocomplete")
async def autocomplete(q: str):
    results = ProductSearch.search(q, 7)
    return results

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        session = SessionMemory()
        SESSIONS[req.session_id] = session
        
    session.push("user", req.text)
    session.extract_facts(req.text)
    
    # 1. Handle Pending Actions (Comparisons)
    if req.pending_action:
        action_type = req.pending_action.get("type")
        prodA = req.pending_action.get("productA")
        
        if action_type == "LOCAL_COMPARE":
            html = ComparisonEngine.generate_local_table(prodA, req.text, session)
            session.push("assistant", "جدول مقارنة محلي")
            def generate_local():
                yield f"data: {json.dumps({'type': 'local', 'html': html, 'is_table': True})}\n\n"
            return StreamingResponse(generate_local(), media_type="text/event-stream")
            
        elif action_type == "AI_COMPARE":
            rA = ProductSearch.search(prodA, 1, session)
            specs = re.sub(r'<[^>]*>', '', str(rA[0].get('specs', '')))[:800] if rA else ""
            sys_prompt = f"أنت خبير مبيعات في العربي. مطلوب مقارنة عادلة.\n[منتجنا]\nالاسم: {rA[0]['name'] if rA else prodA}\nالسعر: {rA[0].get('price') if rA else ''}\nالمواصفات: {specs}\n\n[المنافس]\nالاسم: {req.text}\n\nارسم جدول مقارنة Markdown."
            return StreamingResponse(stream_gemini(sys_prompt, req.text, session, ignore_history=True), media_type="text/event-stream")

    # 2. Detect Intent & Handle Locally First
    intent = IntentDetector.detect(req.text)
    if intent in ["PRICE", "SPECS", "IMAGES", "AVAILABILITY", "WARRANTY"]:
        local_res = LocalEngine.generate(intent, req.text, session)
        if local_res:
            session.push("assistant", "رد محلي")
            def generate_static():
                yield f"data: {json.dumps(local_res)}\n\n"
            return StreamingResponse(generate_static(), media_type="text/event-stream")

    # 3. Handle Persuasion or General Knowledge via AI
    if intent == "PERSUADE":
        hits = ProductSearch.search(req.text, 1, session)
        ctx = ProductSearch.build_context(hits)
        sys_prompt = f"أنت بياع مصري شاطر في شركة العربي. أقنع الزبون بشراء هذا المنتج واذكر المميزات في نقاط:\n\n{ctx}"
        return StreamingResponse(stream_gemini(sys_prompt, req.text, session, ignore_history=True, temperature=0.7), media_type="text/event-stream")
        
    # General AI Fallback with RAG
    hits = ProductSearch.search(req.text, 3, session)
    ctx = ProductSearch.build_context(hits)
    sys_prompt = f"أنت 'العربى' مساعد مبيعات ذكي. اعتمد حصراً على هذه البيانات (RAG):\n{ctx}\n\nإذا لم تجد المعلومة، اعتذر بلباقة."
    
    return StreamingResponse(stream_gemini(sys_prompt, req.text, session, ignore_history=False), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
