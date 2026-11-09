from langchain_core.documents import Document
from typing import List, Dict, Any #类型注解系统
from pathlib import Path
import uuid
from langchain_text_splitters import MarkdownHeaderTextSplitter
import logging
import hashlib

#获取一个以当前模块名为标识的日志记录器,用于打印信息
logger = logging.getLogger(__name__)

class DataPreparationModule:
    '''
    数据准备模块，负责数据的加载，清洗，预处理
    '''
    def __init__(self,data_path:str) -> None:
        '''
        初始化，参数为文件的路径
        我们使用父子文档的方式进行处理
        检索时使用子文档，保持精确性与效率
        生成时使用对应子文档的父文档，保持内容的完整性
        Args:
            data_path: 数据文件夹路径
        '''
        self.data_path = data_path
        self.documents : List[Document] = [] #父文档
        self.chunks : List[Document] = [] #子文档
        self.parent_child_map : Dict[str,str] = {} #父子文档的hash映射
        self.kg: Dict[str, Any] = {
            "dishes": {},
            "ingredient_index": {},
            "technique_index": {},
            "category_index": {},
            "difficulty_index": {},
            "flavor_index": {}
        }

    def load_documents(self) -> List[Document]:
        '''
        载入文档
        Returns:
            加载的文档列表
        '''
        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"): #递归的读取路径下的所有md文件
            #读取文件内容
            with open(md_file,'r',encoding='utf-8') as f:
                content = f.read()

            #为父文档分配唯一id
            parent_id = hashlib.md5(str(md_file).encode()).hexdigest()

            #创建document对象，document是一个带有“内容”和“元数据”的结构化文档对象
            #设计了三项元数据，分别是文件路径，id，文件类型(父文档/子文档)
            doc = Document(
                page_content = content,
                metadata = {
                    "source":str(md_file),
                    "parent_id" :parent_id,
                    "doc_type":"parent"
                }
            )
            documents.append(doc)
        
        for doc in documents:
            self._enhance_metadata(doc)
        self.documents = documents

        return documents

    def _enhance_metadata(self,doc: Document): #用_开头表示内部使用，不被外部调用
        '''
        增强文档的元数据
        通过文档的路径，补充以下元数据
        1.菜品的分类 'category'
        2.菜品的名称 'dish_name'
        3.菜品难度等级 'difficulty'
        Args:
            需要增强的文档
        '''
        #获取文件的路径
        file_path = Path(doc.metadata.get('source','')) #用get函数获取键值，第二个参数为默认值

        #获得这个路径的每一级目录结构 
        #例如"data/articles/article1.md" ->('data', 'articles', 'article1.md')
        path_parts = file_path.parts

        #提取菜品的分类
        category_mapping = {
            'meat_dish': '荤菜', 'vegetable_dish': '素菜', 'soup': '汤品',
            'dessert': '甜品', 'breakfast': '早餐', 'staple': '主食',
            'aquatic': '水产', 'condiment': '调料', 'drink': '饮品'
        }

        #根据文件的路径，在哪个文件夹里，来获得分类
        #先统一初始化为其他
        doc.metadata['category'] = '其他'
        for k,v in category_mapping.items():
            if k in path_parts:
                doc.metadata['category'] = v
                break
        
        #提取菜品的名称
        doc.metadata['dish_name'] = file_path.stem

        #提取菜品难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等难度'
        elif '★★' in content:
            doc.metadata['difficulty'] = '比较简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '简单'
        else:
            doc.metadata['difficulty'] = '未知难度'

    def chunk_documents(self)->List[Document]:
        '''
        基于Markdown结构感知进行子文档分块
        并且添加对应的元数据
        Returns:
            分块后的子文档
        '''
        #如果父文档未加载，抛出异常
        if not self.documents:
            raise ValueError("未加载文档")
    
        #切割子文档
        chunks = self._markdown_header_split()

        #为子文档添加元数据
        #'chunk_id' 'chunk_size'
        for i,chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        #复制值到self去
        self.chunks = chunks
        return chunks

    def build_kg(self) -> Dict[str, Any]:
        import re
        if not self.documents:
            return self.kg
        tech_set = {
            "切","炒","煮","蒸","焖","烤","煎","拌","爆炒","清蒸","水煮","腌制","焯水","烘烤","煮沸","搅拌","泼油"
        }
        flavor_set = {
            "香辣","麻辣","酸甜","清淡","蒜香","葱香","胡椒香","黄油","咖喱"
        }
        for doc in self.documents:
            dish = doc.metadata.get("dish_name", "")
            category = doc.metadata.get("category", "其他")
            difficulty = doc.metadata.get("difficulty", "未知难度")
            text = doc.page_content
            lines = text.splitlines()
            section = ""
            ings = set()
            techs = set()
            flavors = set()
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith("## "):
                    title = line_stripped[3:].strip()
                    if "必备原料" in title or "原料" in title or "工具" in title:
                        section = "ingredients"
                    elif "操作" in title:
                        section = "ops"
                    else:
                        section = ""
                    continue
                if section == "ingredients":
                    if line_stripped.startswith("-") or line_stripped.startswith("•"):
                        name = line_stripped.lstrip("-•").strip()
                        name = re.sub(r"（.*?）", "", name)
                        name = re.sub(r"\(.*?\)", "", name)
                        name = re.sub(r"\d+\s*[gG克个毫升mlML]+", "", name)
                        name = re.sub(r"[，,。;；]", " ", name).strip()
                        if name:
                            parts = re.split(r"\s+", name)
                            if parts:
                                ings.add(parts[0])
                elif section == "ops":
                    for t in list(tech_set):
                        if t in line_stripped:
                            techs.add(t)
                # 风味在全文匹配
                for f in list(flavor_set):
                    if f and f in line_stripped:
                        flavors.add(f)
            self.kg["dishes"][dish] = {
                "ingredients": sorted(list(ings)),
                "techniques": sorted(list(techs)),
                "category": category,
                "difficulty": difficulty,
                "flavors": sorted(list(flavors))
            }
            self.kg["category_index"].setdefault(category, set()).add(dish)
            self.kg["difficulty_index"].setdefault(difficulty, set()).add(dish)
            for ing in ings:
                self.kg["ingredient_index"].setdefault(ing, set()).add(dish)
            for t in techs:
                self.kg["technique_index"].setdefault(t, set()).add(dish)
            for f in flavors:
                self.kg["flavor_index"].setdefault(f, set()).add(dish)
        return self.kg

    def write_kg_html_for_dishes(
        self,
        dishes: List[str],
        output_file: str,
        *,
        min_degree: int = 2,
        topk_edges: int = 600,
        label_degree: int = 6
    ):
        import json
        from pathlib import Path

        if not self.kg.get("dishes"):
            self.build_kg()

        # 兜底，避免空
        if not dishes:
            pool = list(self.kg.get("dishes", {}).keys()) or []
            dishes = pool[:8]

        # ---------- 数据 ----------
        nodes, links, seen = [], [], set()
        for dish in dishes:
            info = self.kg["dishes"].get(dish, {})
            did = f"dish:{dish}"
            if did not in seen:
                nodes.append({"id": did, "label": dish, "type": "dish"}); seen.add(did)

            cat_label = info.get("category") or "其他"
            dif_label = info.get("difficulty") or "未知难度"
            cid, fid = f"cat:{cat_label}", f"dif:{dif_label}"
            if cid not in seen:
                nodes.append({"id": cid, "label": cat_label, "type": "category"}); seen.add(cid)
            if fid not in seen:
                nodes.append({"id": fid, "label": dif_label, "type": "difficulty"}); seen.add(fid)
            links += [{"source": did, "target": cid, "kind": "meta"},
                    {"source": did, "target": fid, "kind": "meta"}]

            for ing in info.get("ingredients", []):
                iid = f"ing:{ing}"
                if iid not in seen:
                    nodes.append({"id": iid, "label": ing, "type": "ingredient"}); seen.add(iid)
                links.append({"source": did, "target": iid, "kind": "ingredient"})
            for tech in info.get("techniques", []):
                tid = f"tech:{tech}"
                if tid not in seen:
                    nodes.append({"id": tid, "label": tech, "type": "technique"}); seen.add(tid)
                links.append({"source": did, "target": tid, "kind": "technique"})

        # 度过滤 + 边采样
        deg = {n["id"]: 0 for n in nodes}
        for e in links:
            s, t = e["source"], e["target"]
            if s in deg and t in deg and s != t:
                deg[s] += 1; deg[t] += 1

        keep_ids = {nid for nid, d in deg.items() if d >= min_degree or nid.startswith("dish:")}
        nodes = [n for n in nodes if n["id"] in keep_ids]
        keep_ids = {n["id"] for n in nodes}
        links = [e for e in links if e["source"] in keep_ids and e["target"] in keep_ids]

        priority = {"ingredient": 0, "technique": 1, "meta": 2}
        links.sort(key=lambda e: priority.get(e["kind"], 9))
        links = links[:topk_edges]

        deg = {n["id"]: 0 for n in nodes}
        for e in links:
            deg[e["source"]] += 1; deg[e["target"]] += 1

        data = {
            "nodes": [{"id": n["id"], "label": n["label"], "type": n["type"], "degree": deg.get(n["id"], 0)} for n in nodes],
            "links": links,
            "label_degree": label_degree,
        }
        data_json = json.dumps(data, ensure_ascii=False)

        # ---------- 交互 HTML ----------
        html_text = """
    <!doctype html>
    <html lang="zh"><meta charset="utf-8">
    <title>ChefGPT KG</title>
    <style>
    :root { --bar-h:48px; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: system-ui,-apple-system,"Segoe UI",Roboto,Arial; }
    #toolbar {
    position:sticky; top:0; z-index:10; display:flex; gap:8px; align-items:center; flex-wrap:wrap;
    padding:8px 10px; border-bottom:1px solid #eee; background:#fff;
    }
    #q{ flex:1; min-width:220px; padding:6px 10px; border:1px solid #ddd; border-radius:6px; }
    .btn{ padding:6px 10px; border:1px solid #ddd; border-radius:6px; background:#f7f7f7; cursor:pointer; }
    #stage { width:100vw; height: calc(100vh - var(--bar-h)); background:#fff; }
    .label { font-size:12px; paint-order: stroke; stroke:#fff; stroke-width:4px; }
    .tip { position:fixed; pointer-events:none; background:rgba(0,0,0,.75); color:#fff; padding:4px 8px; border-radius:6px; font-size:12px; transform:translate(8px,-8px); display:none; }
    #evlog { color:#999; font-size:12px; }
    </style>

    <div id="toolbar">
    <input id="q" placeholder="搜索节点（回车定位）">
    <button id="zoomIn"  class="btn">放大 +</button>
    <button id="zoomOut" class="btn">缩小 −</button>
    <button id="reset"   class="btn">复位</button>
    <span id="evlog">事件：尚无</span>
    </div>

    <div id="stage">
    <svg id="svg" style="width:100%;height:100%;touch-action:none">
        <rect id="bg" x="0" y="0" width="100%" height="100%" fill="transparent" pointer-events="all"></rect>
    </svg>
    </div>
    <div id="tip" class="tip"></div>

    <script>
    const DATA = __DATA_JSON__;
    const COLORS = {dish:"#1f77b4",ingredient:"#2ca02c",technique:"#ff7f0e",category:"#9467bd",difficulty:"#8c564b",other:"#7f7f7f"};
    const LINK_COLOR = "rgba(120,120,120,.35)";
    const svg = document.getElementById('svg');
    const bg  = document.getElementById('bg');
    const tip = document.getElementById('tip');
    const evlog = document.getElementById('evlog');
    function logEv(s){ evlog.textContent = "事件：" + s; }

    let W = svg.clientWidth, H = svg.clientHeight;
    let view = { x:0, y:0, w:W, h:H };
    function applyView(){ svg.setAttribute('viewBox', view.x+" "+view.y+" "+view.w+" "+view.h); }
    applyView();

    const gLinks = document.createElementNS("http://www.w3.org/2000/svg","g");
    const gNodes = document.createElementNS("http://www.w3.org/2000/svg","g");
    svg.appendChild(gLinks); svg.appendChild(gNodes);

    const TYPES = ["dish","ingredient","technique","category","difficulty"];
    const R = Math.min(W,H)/2 - 60;
    const CENTER = { x: W/2, y:H/2 };
    const RING = { dish:R*0.25, ingredient:R*0.55, technique:R*0.75, category:R*0.92, difficulty:R*0.98 };
    const bucket = {}; TYPES.forEach(t => bucket[t]=[]);
    DATA.nodes.forEach(n => (bucket[n.type]||(bucket[n.type]=[])).push(n));
    function polar(cx,cy,r,i,n){ const a=2*Math.PI*(i/Math.max(1,n)); return {x:cx+r*Math.cos(a), y:cy+r*Math.sin(a)}; }

    const POS = {};
    TYPES.forEach(t=>{
    const arr=bucket[t]||[], r=RING[t]||R*0.85;
    for(let i=0;i<arr.length;i++){
        const p=polar(CENTER.x,CENTER.y,r,i,arr.length);
        POS[arr[i].id]={ x:p.x+(Math.random()-.5)*8, y:p.y+(Math.random()-.5)*8 };
    }
    });

    const linkEls = []; const neighborLines = {};
    DATA.links.forEach(e=>{
    const s=POS[e.source], t=POS[e.target]; if(!s||!t) return;
    const L=document.createElementNS("http://www.w3.org/2000/svg","line");
    L.setAttribute("x1",s.x); L.setAttribute("y1",s.y);
    L.setAttribute("x2",t.x); L.setAttribute("y2",t.y);
    L.setAttribute("stroke",LINK_COLOR); L.setAttribute("stroke-width","1");
    gLinks.appendChild(L);
    linkEls.push({el:L, sId:e.source, tId:e.target});
    (neighborLines[e.source]||(neighborLines[e.source]=[])).push(L);
    (neighborLines[e.target]||(neighborLines[e.target]=[])).push(L);
    });

    const nodeIndex = {}; const labelDegree = DATA.label_degree || 6;
    function short(s){ s=String(s||""); return (s.length>8)?(s.slice(0,8)+"…"):s; }

    let draggingNode = null;

    DATA.nodes.forEach(n=>{
    const p=POS[n.id]; if(!p) return;
    const G=document.createElementNS("http://www.w3.org/2000/svg","g");
    const C=document.createElementNS("http://www.w3.org/2000/svg","circle");
    const r=Math.min(18, 8 + Math.max(0,n.degree));
    C.setAttribute("cx",p.x); C.setAttribute("cy",p.y); C.setAttribute("r",r);
    C.setAttribute("fill",COLORS[n.type]||COLORS.other);
    C.setAttribute("stroke","#333"); C.setAttribute("stroke-width","1");
    G.appendChild(C);

    const T=document.createElementNS("http://www.w3.org/2000/svg","text");
    T.setAttribute("x", p.x + r + 4); T.setAttribute("y", p.y + 4);
    T.setAttribute("class","label"); T.setAttribute("fill","#111");
    T.textContent = short(n.label);
    if(n.degree < labelDegree) T.style.display = "none";
    G.appendChild(T);

    G.addEventListener("mousemove",(ev)=>{ tip.style.display="block"; tip.textContent=n.label; tip.style.left=(ev.clientX+10)+"px"; tip.style.top=(ev.clientY-10)+"px"; });
    G.addEventListener("mouseleave",()=> tip.style.display="none");

    let last={x:0,y:0};
    G.addEventListener("mousedown",(e)=>{ draggingNode = n.id; last={x:e.clientX,y:e.clientY}; e.stopPropagation(); logEv("mousedown@node:"+n.id); });
    window.addEventListener("mouseup",()=>{ if(draggingNode){ logEv("mouseup@node"); } draggingNode=null; });
    window.addEventListener("mousemove",(e)=>{
        if(draggingNode!==n.id) return;
        const dx=(e.clientX-last.x)*(view.w/W), dy=(e.clientY-last.y)*(view.h/H);
        last={x:e.clientX,y:e.clientY};
        POS[n.id].x += dx; POS[n.id].y += dy;
        C.setAttribute("cx", POS[n.id].x); C.setAttribute("cy", POS[n.id].y);
        T.setAttribute("x", POS[n.id].x + r + 4); T.setAttribute("y", POS[n.id].y + 4);
        (neighborLines[n.id]||[]).forEach(L=>{
        const it = linkEls.find(o=>o.el===L);
        if(it){ if(it.sId===n.id){ L.setAttribute("x1",POS[n.id].x); L.setAttribute("y1",POS[n.id].y); }
                if(it.tId===n.id){ L.setAttribute("x2",POS[n.id].x); L.setAttribute("y2",POS[n.id].y); } }
        });
        scheduleCull();
    });

    gNodes.appendChild(G);
    nodeIndex[n.id] = { g:G, c:C, t:T, r:r, data:n };
    });

    let panning=false, pan0={x:0,y:0};
    function startPan(e){ if(draggingNode) return; panning=true; pan0={x:e.clientX,y:e.clientY}; logEv("mousedown@svg"); }
    function movePan(e){
    if(!panning || draggingNode) return;
    const dx=(e.clientX-pan0.x)*(view.w/W), dy=(e.clientY-pan0.y)*(view.h/H);
    view.x -= dx; view.y -= dy; pan0={x:e.clientX,y:e.clientY}; applyView();
    }
    function endPan(){ if(panning) logEv("mouseup@svg"); panning=false; }
    svg.addEventListener("mousedown", startPan, true);
    bg .addEventListener("mousedown", startPan, false);
    window.addEventListener("mousemove", movePan, true);
    window.addEventListener("mouseup",   endPan,  true);

    function wheelZoomAt(clientX, clientY, deltaY){
    const scale = (deltaY>0)?1.1:0.9;
    const rect = svg.getBoundingClientRect();
    const mx = view.x + ((clientX - rect.left) / W) * view.w;
    const my = view.y + ((clientY - rect.top)  / H) * view.h;
    view.w *= scale; view.h *= scale;
    view.x = mx - ((clientX - rect.left) / W) * view.w;
    view.y = my - ((clientY - rect.top)  / H) * view.h;
    applyView(); updateLabelByZoom();
    }
    function wheelHandler(e){ e.preventDefault(); wheelZoomAt(e.clientX, e.clientY, e.deltaY); logEv("wheel@svg"); }
    svg.addEventListener("wheel", wheelHandler, {capture:true, passive:false});
    bg .addEventListener("wheel", wheelHandler, {passive:false});

    document.getElementById('zoomIn').onclick  = ()=>{ wheelZoomAt(svg.getBoundingClientRect().left+W/2, svg.getBoundingClientRect().top+H/2, -1); logEv("zoomIn@btn"); };
    document.getElementById('zoomOut').onclick = ()=>{ wheelZoomAt(svg.getBoundingClientRect().left+W/2, svg.getBoundingClientRect().top+H/2,  1); logEv("zoomOut@btn"); };

    document.getElementById('reset').addEventListener('click',()=>{
    view={x:0,y:0,w:W,h:H}; applyView(); updateLabelByZoom(true); logEv("reset@btn");
    });

    document.getElementById('q').addEventListener('keydown',(e)=>{
    if(e.key!=='Enter') return;
    const q=(e.target.value||'').trim(); if(!q) return;
    const hit = DATA.nodes.find(n=> (n.label&&n.label.indexOf(q)>=0) || (n.id&&n.id.indexOf(q)>=0));
    if(!hit) return alert('未找到：'+q);
    const p=POS[hit.id]; if(!p) return;
    const w=view.w*0.6, h=view.h*0.6;
    view={ x:p.x-w/2, y:p.y-h/2, w, h }; applyView(); updateLabelByZoom(true); logEv("search:"+q);
    });

    function updateLabelByZoom(){
    const zoom = W / view.w;
    for(const id in nodeIndex){
        const o=nodeIndex[id], d=o.data.degree||0;
        if(d >= __LABEL_DEGREE__) o.t.style.display="block";
        else o.t.style.display = (zoom >= 1.3) ? "block" : "none";
    }
    scheduleCull();
    }
    let cullScheduled=null;
    function scheduleCull(){ if(cullScheduled) cancelAnimationFrame(cullScheduled); cullScheduled=requestAnimationFrame(cullOverlaps); }
    function cullOverlaps(){
    const shown = Object.values(nodeIndex).map(o=>o.t).filter(t=>t.style.display!=="none");
    const items = shown.map(t=>({ el:t, box:t.getBBox(), deg: parseInt(t.__deg||0) }));
    items.sort((a,b)=> a.box.y - b.box.y);
    for(let i=0;i<items.length;i++){
        for(let j=i+1;j<items.length;j++){
        const A=items[i].box, B=items[j].box;
        if(B.y > A.y + A.height + 2) break;
        const overlap = !(B.x > A.x + A.width + 4 || A.x > B.x + B.width + 4);
        if(overlap){
            const aD=items[i].deg, bD=items[j].deg;
            if(aD===bD){ (items[i].box.width <= items[j].box.width) ? (items[j].el.style.display="none") : (items[i].el.style.display="none"); }
            else { (aD >= bD) ? (items[j].el.style.display="none") : (items[i].el.style.display="none"); }
        }
        }
    }
    }
    for(const id in nodeIndex){ nodeIndex[id].t.__deg = (nodeIndex[id].data.degree||0); }
    updateLabelByZoom();

    new ResizeObserver(()=>{
    W = svg.clientWidth; H = svg.clientHeight;
    const cx = view.x + view.w/2, cy = view.y + view.h/2;
    view.w = W; view.h = H; view.x = cx - W/2; view.y = cy - H/2;
    applyView();
    }).observe(document.getElementById('stage'));
    </script>
    </html>
    """.replace("__DATA_JSON__", data_json).replace("__LABEL_DEGREE__", str(label_degree))

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(html_text, encoding="utf-8")



    def _markdown_header_split(self) -> List[Document]:
        '''
        使用Markdown标题分割器进行结构化分割
        '''
        #定义要分割的标题层级
        headers_to_split_on = [
            ("#","主标题"), #菜品名称
            ("##","二级标题"), #必备原料、计算、操作等
            ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        #创建Markdown分割器
        #这个langchain的Markdown分割器会将每一级标题下的内容作为文本块
        #同时将层级信息作为元数据附加上去
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        all_chunks = []
        for doc in self.documents:
            try:
                #对每个文档进行Markdown分割
                single_chunks = markdown_splitter.split_text(doc.page_content)

                #建立父子文档关系
                #先拿到大的父文档的id
                parent_id = doc.metadata["parent_id"]

                #遍历所有的子块
                for i,chunk in enumerate(single_chunks):
                    #获取一个子文档的标识
                    child_id = str(uuid.uuid4())

                    #先把父文档的元数据拷贝给子文档
                    chunk.metadata.update(doc.metadata)
                    #赋值dish_name
                    chunk.metadata["dish_name"] = doc.metadata.get("dish_name", "未知菜品")
                    #这里我们复制了"source"，"parent_id"，"doc_type"，'category'，'dish_name'，'difficulty'
                    #我们需要更改其中的"doc_type"，并添加一些元数据
                    chunk.metadata.update(
                        {
                            "chunk_id":child_id,
                            "parent_id":parent_id,  
                            "doc_type":"child",#标记为子文档
                            "chunk_index":i #子文档在父文档中的位置
                        
                        }
                    )

                    #再建立父文档子文档的映射关系
                    #因为我们需要根据子文档查找父文档，所以用子文档id作为键,父文档作为值
                    self.parent_child_map[child_id] = parent_id
                #把这一个文件的所有块加入块集合中
                all_chunks.extend(single_chunks)
            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)

        return all_chunks

    def get_parent_document(self,child_chunks:List[Document])->List[Document]:
        '''
        根据子文档查找父文档
        由于找到的子文档可能有多个属于同一个父文档，会导致返回多个重复的父文档
        所以我们必须对重复的父文档进行去重
        Args:
            需要查找父文档的子文档
        Returns:
            找到的父文档
        '''
        #我们先初步收集父文档，没有排序
        parent_doc_map = {}
        #我们再统计父文档被映射的次数，代表了父文档的重要性
        parent_relevance = {}

        for i, doc in enumerate(child_chunks):
            print(f"chunk {i}: dish={doc.metadata.get('dish_name')}, parent_id={doc.metadata.get('parent_id')}, source={doc.metadata.get('source')}")

        #遍历所有的子文档，统计父文档与相关性次数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                #增加相关性次数
                parent_relevance[parent_id] = parent_relevance.get(parent_id,0)+1

                #缓存父文档
                if parent_id not in parent_doc_map:
                    #检测下这个父文档是否是我们加载的父文档
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_doc_map[parent_id] = doc
                            break

        #根据相关性排序排序,降序,越在前的文档越重要
        sorted_parent_ids = sorted(parent_relevance.keys(),
                            key = lambda x:parent_relevance[x],reverse=True
        )

        #构建最后要返回的父文档
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_doc_map:
                parent_docs.append(parent_doc_map[parent_id])
        
         # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs

        return parent_docs





            









        
    

        


        
