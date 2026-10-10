'use strict';
const el = id => document.getElementById(id);
const names = {fmcib:'FMCIB',vista:'VISTA3D'};
const labels = {id:'Feature 编号',maximum:'最大响应',lung_enrichment:'肺部富集程度',liver_enrichment:'肝脏富集程度',outside_fraction:'图像边缘空气响应',high_hu_fraction:'高 HU 区域的响应'};
const memories = JSON.parse(localStorage.getItem('sclvmi-heatmap-selections')||'{}');
const state = {catalog:[],run:null,scope:'whole',feature:0,case:0,mode:'spatial',panel:'explore',point:null,pointKey:null,meta:null,features:[],cases:[],page:0,request:0,space:false,pan:{},drag:null,rects:{},scaleKey:null,pending:false};
const pageSize=80,volumeCache=new Map();
let renderer,controller,drawFrame=0,findingsRequest=0;
const format=value=>Number(value).toLocaleString(ViewerLanguage.locale,{maximumFractionDigits:3});
const text=value=>ViewerLanguage.value==='en'?ViewerLanguage.text(value):value;
const node=(tag,value,className)=>{const item=document.createElement(tag);if(value!=null)item.textContent=value;if(className)item.className=className;return item;};
const spec=()=>state.catalog.find(row=>row.run_id===state.run);
const selectionKey=()=>`${state.scope}/${state.run}`;
const query=()=>new URLSearchParams({run:state.run,scope:state.scope,case:state.case,feature:state.feature,mode:state.mode});

async function get(url,signal,binary=false){
  if(binary&&volumeCache.has(url)){const value=volumeCache.get(url);volumeCache.delete(url);volumeCache.set(url,value);return value;}
  const response=await fetch(url,{signal});
  if(!response.ok){const data=await response.json();throw new Error(data.detail||`HTTP ${response.status}`);}
  const value=binary?new Float32Array(await response.arrayBuffer()):await response.json();signal?.throwIfAborted();
  if(value.translations)ViewerLanguage.add(value.translations);
  if(binary){volumeCache.set(url,value);let bytes=[...volumeCache.values()].reduce((sum,item)=>sum+item.byteLength,0);for(const [key,item] of volumeCache){if(bytes<=160*1024*1024)break;volumeCache.delete(key);bytes-=item.byteLength;}}
  return value;
}
function remember(){memories[selectionKey()]={feature:state.feature,mode:state.mode};memories[`case:${state.scope}`]=state.case;localStorage.setItem('sclvmi-heatmap-selections',JSON.stringify(memories));}
function syncURL(push=false){
  const values=query();if(state.panel==='findings'){values.set('panel','findings');values.set('findings',el('findings-scope').value);}
  const url=`/whole?${values}`;
  if(push&&`${location.pathname}${location.search}`!==url)history.pushState(null,'',url);else history.replaceState(null,'',url);
  localStorage.setItem('sclvmi-heatmap-location',url);remember();
}
function loading(message,error=false){
  const shell=document.querySelector('.viewer-shell');shell.setAttribute('aria-busy',String(!error));shell.toggleAttribute('data-error',error);
  el('viewer-loading').hidden=false;el('viewer-loading-text').textContent=text(message);document.querySelector('.viewer-grid').style.visibility='hidden';state.meta=null;state.rects={};
  for(const axis of [0,1,2]){const canvas=el(`canvas-${axis}`);canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);el(`slice-${axis}`).disabled=true;el(`position-${axis}`).textContent='';}
  for(const id of ['peak','center','strongest','reset-scale'])el(id).disabled=true;
  for(const id of ['map-stats','point-value','map-label','geometry-note','feature-evidence','case-label'])el(id).replaceChildren();
}
function report(error){if(error.name==='AbortError')return;el('loading').textContent='';el('error-message').textContent=error.message;el('error').hidden=false;loading('影像读取失败。可以重新读取，或选择另一份图像。',true);}
function run(action){Promise.resolve().then(action).catch(report);}
function modes(){
  if(spec().site==='global')return [['global',state.scope==='whole'?'每个输入 crop 的响应':'整个 crop 的响应']];
  return state.scope==='whole'?[['spatial','空间激活 · 原始采样'],['single','每个区域 · 中央位置'],['mean','每个区域 · 平均值'],['maximum','每个区域 · 峰值']]:[['spatial','空间激活 · 插值显示'],['native','空间激活 · 原始采样']];
}
function controls(){
  const current=spec(),families=[...new Set(state.catalog.filter(row=>row.scopes.includes(state.scope)).map(row=>row.family))];
  for(const option of el('category').options)option.textContent=text(labels[option.value]);
  el('model').replaceChildren();
  for(const reference of [false,true]){
    const group=node('optgroup');group.label=text(reference?'研究观察使用的字典':'选定 SAE 组合');
    for(const family of families){const row=state.catalog.find(item=>item.family===family);if(row.reference!==reference)continue;const option=node('option',`${names[row.model]} · ${row.site} · k${row.k}`);option.value=family;group.append(option);}
    if(group.children.length)el('model').append(group);
  }
  el('model').value=current.family;
  el('seed').replaceChildren(...state.catalog.filter(row=>row.family===current.family).map(row=>{const option=node('option',row.seed);option.value=row.run_id;return option;}));
  el('seed').value=state.run;el('seed').disabled=el('seed').options.length===1;el('scope').value=state.scope;
  const available=modes();if(!available.some(([value])=>value===state.mode))state.mode=available[0][0];
  el('mode').replaceChildren(...available.map(([value,label])=>{const option=node('option',text(label));option.value=value;return option;}));el('mode').value=state.mode;
  el('dictionary-stamp').textContent=`${names[current.model]} · ${current.site} · seed ${current.seed}`;
  el('dictionary-note').textContent=`${format(current.features)} features · k${current.k} · seed ${current.seed}. ${text('Feature 编号属于当前字典。')}`;
  el('feature-title').textContent=`${names[current.model]} · ${current.site} · Feature ${state.feature}`;el('feature-description').textContent=state.features.find(row=>row.id===state.feature)?.description||'';
  el('cases-title').textContent=text(state.scope==='whole'?'完整 CT 病例':'结节 crop 病例');
  el('cases-note').textContent=text(state.scope==='whole'?'每组第一例用于观察，第二例用于确认':'128 名不同患者的开发集输入');
  el('dataset-note').textContent=state.scope==='whole'?'PETWB · training':'LUNA25 · development';el('source-note').textContent=current.reference?'20261008 · '+text('研究观察字典'):'20261009 · '+text('选定 SAE 组合');
  el('scope-help').textContent=text(current.reference?'研究观察中的 FMCIB 使用连续卷积，VISTA3D 使用固定窗口。具体采样方式随字典身份显示。':state.scope==='whole'?'选定字典使用与训练相同大小的输入窗口。FMCIB 每隔 32 mm、VISTA3D 每隔 48 mm 移动窗口；空间图保留中央位置，global 图显示每个窗口的响应。窗口上下文可能影响相邻区域的响应。':'空间图可以选择原始采样或插值显示。global 给出整个 crop 的单个响应，CT 仍可联动浏览。');
  el('dictionary-checks').hidden=current.reference;
  if(!current.reference){
    const rows=[['开发集 FVU',current.development.fvu.toFixed(5)],['平均 cosine',current.development.cosine.toFixed(5)],['分类概率 MAE',current.downstream.probability_mae.toFixed(5)],['重建后 AUROC',current.downstream.recovered_auroc.toFixed(5)],['未激活 Feature 比例',`${(current.development.inactive_fraction*100).toFixed(2)}%`]];
    el('quality-values').replaceChildren(...rows.flatMap(([label,value])=>[node('dt',text(label)),node('dd',value)]));
    el('quality-scope').textContent=text('这些检查来自 LUNA25 开发集。完整 CT 响应属于跨数据来源的探索；医学含义和具体 Feature 的干预效果需要独立证据。');
    el('dictionary-identity').textContent=`${current.run_id} · SHA-256 ${current.dictionary_sha256}`;
  }
}
function featureMetric(row,key){
  if(Number.isFinite(row[key]))return {rank:row[key],label:format(row[key])};
  const region=key==='lung_enrichment'?'lung':key==='liver_enrichment'?'liver':null;
  if(region&&row.discovery_region_means?.[region]>0)return {rank:Infinity,label:text('参考区域均值为零')};
  return {rank:-Infinity,label:text(row[key]===null?(region?'两组均无响应':key==='maximum'?'准备中':'观察病例无响应'):'统计中')};
}
function featureRows(){
  const term=el('search').value.trim().toLowerCase(),key=el('category').value;
  return state.features.filter(row=>`${row.id} ${row.description||''} ${ViewerLanguage.text(row.description||'')}`.toLowerCase().includes(term)).sort((a,b)=>key==='id'?a.id-b.id:featureMetric(b,key).rank-featureMetric(a,key).rank||a.id-b.id);
}
function featureList(locate=false){
  const rows=featureRows();if(locate){const index=rows.findIndex(row=>row.id===state.feature);if(index>=0)state.page=Math.floor(index/pageSize);}
  const pages=Math.max(1,Math.ceil(rows.length/pageSize));state.page=Math.max(0,Math.min(state.page,pages-1));
  const list=el('feature-list'),scroll=list.scrollTop,focused=document.activeElement?.dataset.feature;
  list.replaceChildren(...rows.slice(state.page*pageSize,(state.page+1)*pageSize).map(row=>{
    const button=node('button',null,`feature-item ${row.id===state.feature?'active':''}`);button.dataset.feature=row.id;button.setAttribute('aria-current',String(row.id===state.feature));button.append(node('strong',row.id),node('span',text(row.description||'Feature')));
    const key=el('category').value;if(key!=='id')button.append(node('small',`${text(labels[key])} ${featureMetric(row,key).label}`));return button;
  }));
  list.scrollTop=scroll;if(focused)list.querySelector(`[data-feature="${focused}"]`)?.focus({preventScroll:true});
  el('feature-count').textContent=format(state.features.length);el('empty-features').hidden=rows.length!==0;el('page-label').textContent=`${state.page+1} / ${pages}`;el('previous-page').disabled=state.page===0;el('next-page').disabled=state.page===pages-1;
  if(locate){const selected=list.querySelector(`[data-feature="${state.feature}"]`);if(selected){const bounds=list.getBoundingClientRect(),item=selected.getBoundingClientRect();if(getComputedStyle(list).display==='flex'){if(item.left<bounds.left)list.scrollLeft-=Math.ceil(bounds.left-item.left);else if(item.right>bounds.right)list.scrollLeft+=Math.ceil(item.right-bounds.right);}else if(item.top<bounds.top)list.scrollTop-=Math.ceil(bounds.top-item.top);else if(item.bottom>bounds.bottom)list.scrollTop+=Math.ceil(item.bottom-bounds.bottom);}}
}
function updateCases(){
  const current=spec();
  const key=`${state.scope}/${current.model}/${ViewerLanguage.value}/${state.cases.map(row=>Number(row.ready)).join('')}`;
  if(el('case-strip').dataset.catalog!==key){
    el('case-select').replaceChildren(...state.cases.map(row=>{const option=node('option',`${row.label} · ${row.split}${row.ready?'':` · ${text('准备中')}`}`);option.value=row.id;return option;}));
    el('case-strip').replaceChildren(...state.cases.map(row=>{const button=node('button',null,'case-tile');button.dataset.case=row.id;const image=node('img');image.className='whole-thumb';image.src=`/api/heatmaps/thumbnail/${state.scope}/${current.model}/${row.id}`;image.alt=`${row.label} ${text('冠状面 CT')}`;image.loading='lazy';button.append(image,node('strong',row.label),node('span',row.split));return button;}));el('case-strip').dataset.catalog=key;
  }
  el('case-select').value=state.case;for(const button of el('case-strip').children){const active=Number(button.dataset.case)===state.case;button.classList.toggle('active',active);button.setAttribute('aria-current',String(active));}
  el('case-label').textContent=`${state.cases[state.case]?.label||''} · ${state.cases[state.case]?.split||''}`;
}
function showMetadata(){
  const meta=state.meta;if(!meta)return;const pooled=Boolean(meta.pooled),spacing=Array.isArray(meta.map_spacing_mm)?meta.map_spacing_mm.map(format).join(' × '):format(meta.map_spacing_mm);
  el('map-label').textContent=pooled?text('整个 crop 的响应'):`${el('mode').selectedOptions[0].textContent} · ${spacing} mm`;el('geometry-note').textContent=pooled?text('整个输入对应一个响应值'):text(meta.scale_source);
  const rows=[['最大值',format(meta.maximum)],['平均值',format(meta.mean)],['有响应的位置',`${format(meta.nonzero_fraction*100)}%`],['真实采样网格',(meta.native_shape||meta.map_shape).join(' × ')]];el('map-stats').replaceChildren(...rows.flatMap(([label,value])=>[node('dt',text(label)),node('dd',value)]));
  el('feature-evidence').textContent=state.features.find(row=>row.id===state.feature)?.evidence||text(pooled?'global 对整个输入给出一个响应值。':meta.window_response?'每个区域的颜色表示对应输入 crop 的 global 响应。':'');
}
async function loadSelection({catalog=false,locate=false,push=false,resetPoint=false,point=null}={}){
  catalog=catalog||!state.features.length;controller?.abort();controller=new AbortController();const signal=controller.signal,id=++state.request,started=performance.now();state.pending=false;
  if(catalog){state.features=[];state.cases=[];el('feature-list').replaceChildren();el('feature-list').inert=true;el('case-select').replaceChildren();el('case-strip').replaceChildren();delete el('case-strip').dataset.catalog;el('feature-count').textContent='';el('page-label').textContent='…';el('previous-page').disabled=true;el('next-page').disabled=true;}
  controls();loading(`正在加载 ${names[spec().model]} · Feature ${state.feature}…`);el('error').hidden=true;el('loading').textContent=text('读取影像与响应…');
  try{
    if(catalog){
      const data=await get(`/api/heatmaps/features?${new URLSearchParams({run:state.run,scope:state.scope})}`,signal);if(id!==state.request)return;state.features=data.features;state.cases=data.cases;
      const order=el('category').value;el('category').replaceChildren(...data.rankings.map(key=>{const option=node('option',text(labels[key]));option.value=key;return option;}));if(data.rankings.includes(order))el('category').value=order;el('feature-list').inert=false;
    }
    if(!Number.isInteger(state.feature)||state.feature<0||state.feature>=spec().features)throw new Error(text('Feature 编号超出当前字典范围。'));
    if(!Number.isInteger(state.case)||state.case<0||state.case>=state.cases.length)throw new Error(text('病例编号超出当前图像范围。'));
    controls();syncURL(push);featureList(locate);updateCases();
    if(!state.cases[state.case].ready){state.pending=true;loading('完整 CT 响应正在准备。',true);el('loading').textContent='';await preparation();return;}
    const selection=query().toString(),current=spec();
    const [meta,ct,activity]=await Promise.all([get(`/api/heatmaps/map?${selection}`,signal),get(`/api/heatmaps/ct/${state.scope}/${current.model}/${state.case}`,signal,true),get(`/api/heatmaps/activity?${selection}`,signal,true)]);if(id!==state.request)return;
    if(!renderer)renderer=new WholeVolumeRenderer();renderer.upload(meta,ct,activity,state.mode==='native'?'spatial':state.mode);state.meta=meta;state.pan={};
    const pointKey=`${state.scope}/${current.model}/${state.case}`;if(resetPoint||state.pointKey!==pointKey||!state.point)state.point=meta.shape.map(size=>Math.floor(size/2));if(point)state.point=[...point];state.point=state.point.map((value,axis)=>Math.max(0,Math.min(meta.shape[axis]-1,value)));state.pointKey=pointKey;
    const scaleKey=`${state.scope}/${state.run}/${state.feature}/${state.mode}`;if(scaleKey!==state.scaleKey){el('scale').value=Number(meta.scale.toPrecision(5));state.scaleKey=scaleKey;}
    for(let axis=0;axis<3;axis++){el(`slice-${axis}`).max=meta.shape[axis]-1;el(`slice-${axis}`).value=state.point[axis];el(`slice-${axis}`).disabled=false;}
    const pooled=Boolean(meta.pooled);for(const name of ['overlay','opacity','threshold','scale','reset-scale','peak'])el(name).disabled=pooled;
    el('center').disabled=false;el('strongest').hidden=state.scope!=='crop';el('strongest').disabled=meta.strongest_case==null;
    showMetadata();
    paintAll();document.querySelector('.viewer-grid').style.visibility='visible';el('viewer-loading').hidden=true;document.querySelector('.viewer-shell').setAttribute('aria-busy','false');el('loading').textContent='';el('run-progress').textContent='';document.querySelector('.viewer-shell').dataset.selection=selection;document.querySelector('.viewer-shell').dataset.loadMs=String(performance.now()-started);
  }catch(error){if(id===state.request)report(error);}
}
function draw(axis){
  const meta=state.meta;if(!meta||!renderer)return;const canvas=el(`canvas-${axis}`),box=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;if(!box.width||!box.height)return;
  const widthPx=Math.round(box.width*dpr),heightPx=Math.round(box.height*dpr);if(canvas.width!==widthPx||canvas.height!==heightPx){canvas.width=widthPx;canvas.height=heightPx;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#0b151a';ctx.fillRect(0,0,box.width,box.height);
  const [u,v]=[0,1,2].filter(value=>value!==axis),nx=meta.shape[u],ny=meta.shape[v],physical=[nx*meta.spacing[u],ny*meta.spacing[v]],zoom=Number(el('zoom').value),unit=Math.min((box.width-32)/physical[0],(box.height-32)/physical[1])*zoom;
  const width=physical[0]*unit,height=physical[1]*unit,pan=state.pan[axis]||[0,0],left=(box.width-width)/2+pan[0],top=(box.height-height)/2+pan[1];state.rects[axis]={left,top,width,height,nx,ny,u,v};
  const windows={tissue:[-160,240],lung:[-1000,400],bone:[-500,1500]},scale=Math.max(.001,Number(el('scale').value)),opacity=Number(el('opacity').value)/100,threshold=Number(el('threshold').value)/100,overlay=el('overlay').checked&&!state.space&&!meta.pooled;
  const off=renderer.render(axis,state.point[axis],{window:windows[el('window').value],scale,opacity:overlay?opacity:0,threshold});ctx.imageSmoothingEnabled=false;ctx.drawImage(off,left,top,width,height);
  const x=left+(state.point[u]+.5)/nx*width,y=top+(ny-state.point[v]-.5)/ny*height;ctx.strokeStyle='rgba(120,223,205,.8)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+height);ctx.moveTo(left,y);ctx.lineTo(left+width,y);ctx.stroke();
  const directions=axis===2?['L','R','A','P']:axis===1?['L','R','S','I']:['P','A','S','I'];ctx.fillStyle='#dce8e8';ctx.font='12px "Segoe UI",sans-serif';ctx.textAlign='center';ctx.fillText(directions[0],12,box.height/2);ctx.fillText(directions[1],box.width-12,box.height/2);ctx.fillText(directions[2],box.width/2,15);ctx.fillText(directions[3],box.width/2,box.height-7);el(`position-${axis}`).textContent=`${state.point[axis]+1} / ${meta.shape[axis]}`;el(`slice-${axis}`).value=state.point[axis];
}
function paintAll(){if(!state.meta)return;const started=performance.now();[0,1,2].forEach(draw);el('legend-high').textContent=format(el('scale').value);const [ct,activity]=renderer.point(state.point);el('point-value').textContent=`${format(ct)} HU · activation ${format(activity)}`;document.querySelector('.viewer-shell').dataset.paintMs=String(performance.now()-started);}
function drawAll(){if(!drawFrame)drawFrame=requestAnimationFrame(()=>{drawFrame=0;run(paintAll);});}
function movePoint(point){if(state.meta){state.point=point;drawAll();}}
function step(axis,delta){if(!state.meta)return;const point=[...state.point];point[axis]=Math.max(0,Math.min(state.meta.shape[axis]-1,point[axis]+delta));movePoint(point);}
function selectFeature(value,focus=false){
  if(!Number.isInteger(value)||value<0||value>=spec().features)return;
  const changed=value!==state.feature;state.feature=value;featureList(true);
  if(focus)el('feature-list').querySelector(`[data-feature="${value}"]`)?.focus({preventScroll:true});
  if(changed)run(()=>loadSelection({locate:true,push:true}));
}
function selectDictionary(runId,scope=state.scope){remember();const changingScope=scope!==state.scope;state.run=runId;state.scope=scope;if(changingScope)state.case=memories[`case:${scope}`]||0;const previous=memories[selectionKey()];state.feature=previous?.feature||0;state.mode=previous?.mode||'spatial';el('search').value='';state.page=0;run(()=>loadSelection({catalog:true,locate:true,push:true}));}
for(const axis of [0,1,2]){
  const canvas=el(`canvas-${axis}`);el(`slice-${axis}`).addEventListener('input',event=>{if(state.meta){const point=[...state.point];point[axis]=Number(event.target.value);movePoint(point);}});
  canvas.addEventListener('wheel',event=>{event.preventDefault();step(axis,event.deltaY>0?1:-1);},{passive:false});canvas.addEventListener('keydown',event=>{if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();step(axis,['ArrowUp','ArrowRight'].includes(event.key)?1:-1);}});
  canvas.addEventListener('pointerdown',event=>{if(!state.rects[axis]||!event.isPrimary)return;canvas.focus({preventScroll:true});if(event.shiftKey){state.drag={axis,pointer:event.pointerId,x:event.clientX,y:event.clientY,pan:state.pan[axis]||[0,0]};canvas.setPointerCapture(event.pointerId);return;}const box=canvas.getBoundingClientRect(),r=state.rects[axis],point=[...state.point];point[r.u]=Math.max(0,Math.min(r.nx-1,Math.floor((event.clientX-box.left-r.left)/r.width*r.nx)));point[r.v]=Math.max(0,Math.min(r.ny-1,r.ny-1-Math.floor((event.clientY-box.top-r.top)/r.height*r.ny)));movePoint(point);});
  canvas.addEventListener('pointermove',event=>{const drag=state.drag;if(drag?.axis!==axis||drag.pointer!==event.pointerId)return;state.pan[axis]=[drag.pan[0]+event.clientX-drag.x,drag.pan[1]+event.clientY-drag.y];drawAll();});for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>{state.drag=null;});canvas.addEventListener('dblclick',()=>{state.pan={};el('zoom').value='1';drawAll();});
}
el('feature-list').onclick=event=>{const button=event.target.closest('[data-feature]');if(button)selectFeature(Number(button.dataset.feature),true);};
el('feature-list').onkeydown=event=>{
  if(!['ArrowUp','ArrowDown'].includes(event.key)||event.altKey||event.ctrlKey||event.metaKey)return;
  event.preventDefault();const rows=featureRows();if(!rows.length)return;
  const index=rows.findIndex(row=>row.id===state.feature),next=index<0?state.page*pageSize:index+(event.key==='ArrowDown'?1:-1);
  selectFeature(rows[Math.max(0,Math.min(rows.length-1,next))].id,true);
};
el('previous-page').onclick=()=>{state.page--;featureList();};el('next-page').onclick=()=>{state.page++;featureList();};el('search').oninput=()=>{state.page=0;featureList();};el('category').onchange=()=>{state.page=0;featureList();};
el('search').onkeydown=event=>{
  if(event.key!=='Enter'||event.isComposing)return;
  event.preventDefault();const term=el('search').value.trim();if(!term)return;
  if(/^\d+$/.test(term)){const value=Number(term);if(value>=spec().features)return;el('search').value='';selectFeature(value,true);}
  else {const first=featureRows()[0];if(first)selectFeature(first.id,true);}
};
el('model').onchange=()=>{const candidates=state.catalog.filter(row=>row.family===el('model').value);selectDictionary((candidates.find(row=>row.seed===spec().seed)||candidates[0]).run_id);};el('seed').onchange=()=>selectDictionary(el('seed').value);
el('scope').onchange=()=>{const scope=el('scope').value,current=spec();const next=current.scopes.includes(scope)?current:state.catalog.find(row=>!row.reference&&row.model===current.model&&row.site===current.site);selectDictionary(next.run_id,scope);};
el('mode').onchange=()=>{state.mode=el('mode').value;run(()=>loadSelection({push:true}));};
function selectCase(value){if(value===state.case)return;state.case=value;run(()=>loadSelection({push:true,resetPoint:true}));}
el('case-select').onchange=()=>selectCase(Number(el('case-select').value));el('case-strip').onclick=event=>{const button=event.target.closest('[data-case]');if(button)selectCase(Number(button.dataset.case));};el('strongest').onclick=()=>{if(state.meta?.strongest_case!=null)selectCase(state.meta.strongest_case);};
for(const id of ['overlay','window','zoom'])el(id).onchange=drawAll;el('scale').onchange=()=>{if(!Number.isFinite(Number(el('scale').value))||Number(el('scale').value)<=0)el('scale').value=state.meta?.scale||1;drawAll();};
for(const id of ['opacity','threshold'])el(id).oninput=()=>{el(`${id}-value`).textContent=`${el(id).value}%`;drawAll();};el('peak').onclick=()=>{if(state.meta)movePoint([...state.meta.peak]);};el('center').onclick=()=>{if(state.meta)movePoint(state.meta.shape.map(size=>Math.floor(size/2)));};el('reset-scale').onclick=()=>{if(state.meta){el('scale').value=Number(state.meta.scale.toPrecision(5));drawAll();}};el('retry').onclick=()=>run(()=>loadSelection({catalog:!state.features.length}));
el('help-toggle').onclick=()=>{if(state.panel!=='explore')showPanel('explore',true);el('help').hidden=!el('help').hidden;el('help-toggle').setAttribute('aria-expanded',String(!el('help').hidden));};
document.querySelectorAll('.expand').forEach(button=>button.onclick=()=>{const grid=document.querySelector('.viewer-grid'),view=button.closest('.viewport'),expanded=!grid.classList.contains('expanded');grid.classList.toggle('expanded',expanded);document.querySelectorAll('.viewport').forEach(item=>item.classList.toggle('selected-view',expanded&&item===view));document.querySelectorAll('.expand').forEach(item=>item.textContent=text(expanded&&item===button?'恢复':'放大'));requestAnimationFrame(drawAll);});
document.addEventListener('keydown',event=>{if(event.code==='Space'&&!['INPUT','SELECT','BUTTON','TEXTAREA','A'].includes(document.activeElement.tagName)){event.preventDefault();state.space=true;drawAll();}});document.addEventListener('keyup',event=>{if(event.code==='Space'){state.space=false;drawAll();}});window.addEventListener('blur',()=>{state.space=false;state.drag=null;drawAll();});new ResizeObserver(drawAll).observe(document.querySelector('.viewer-grid'));
function showPanel(panel,push=false,sync=true){state.panel=panel;const findings=panel==='findings';el('explore-panel').hidden=findings;el('findings-panel').hidden=!findings;for(const [id,active] of [['explore-tab',!findings],['findings-tab',findings]]){el(id).classList.toggle('active',active);if(active)el(id).setAttribute('aria-current','page');else el(id).removeAttribute('aria-current');}if(sync)syncURL(push);if(!findings)requestAnimationFrame(drawAll);}
el('explore-tab').onclick=()=>showPanel('explore',true);el('home-link').onclick=event=>{event.preventDefault();showPanel('explore',true);};el('findings-tab').onclick=()=>{showPanel('findings',true);run(loadFindings);};el('findings-scope').onchange=()=>{syncURL(true);run(loadFindings);};
async function loadFindings(){
  const id=++findingsRequest,scope=el('findings-scope').value;el('findings-list').textContent=text('读取研究观察…');const data=await get(scope==='whole'?'/api/whole/findings':'/api/catalog');if(id!==findingsRequest)return;const items=scope==='whole'?data.findings:data.findings.items;
  el('findings-list').replaceChildren(...items.map(item=>{
    const article=node('article',null,'finding'),body=node('div'),links=node('div',null,'finding-links');body.append(node('h2',text(item.title)),node('p',text(item.observation)),node('p',text(item.evidence),'finding-evidence'),node('p',text(item.boundary),'qualification'));
    for(const link of scope==='whole'?(item.links||[]):item.features.map(feature=>({feature}))){const button=node('button',scope==='whole'?`${names[link.model]} · ${link.feature} · CT ${String(link.case+1).padStart(2,'0')}`:`Feature ${link.feature}`);button.onclick=()=>{if(scope==='crop'){location.href=`/experiments?${new URLSearchParams(link)}`;return;}run(async()=>{remember();const target=state.catalog.find(row=>row.reference&&row.model===link.model);state.run=target.run_id;state.scope='whole';state.feature=link.feature;state.case=link.case;state.mode=link.mode||'spatial';el('search').value='';showPanel('explore',false,false);await loadSelection({catalog:true,locate:true,push:true,resetPoint:true,point:link.point});});};links.append(button);}
    article.append(body,links);if(item.figure){const figure=node('figure'),image=node('img');image.src=item.figure;image.alt=text(item.title);figure.append(image,node('figcaption',text(item.caption||'')));article.append(figure);}return article;
  }));
}
async function preparation(){
  if(!state.pending)return;const id=state.request,model=spec().model,data=await get(`/api/heatmaps/preparation?run=${encodeURIComponent(state.run)}`);if(id!==state.request||!state.pending)return;const item=data[model];
  if(item.state==='running')el('run-progress').textContent=`${names[model]} · CT ${item.case+1}/${item.cases} · ${format(item.completed)}/${format(item.total)} ${text('窗口')} · ${format(item.tiles_per_second)} ${text('窗口/秒')}`;
  else el('run-progress').textContent=text(item.state==='paused'?'完整 CT 响应计算已暂停。':'完整 CT 响应正在准备。');if(item.dictionary_ready||item.state==='completed')await loadSelection({catalog:true});
}
setInterval(()=>run(preparation),15000);
function readLocation(){const params=new URLSearchParams(location.search),old=params.get('model');state.run=params.get('run')||(old?state.catalog.find(row=>row.reference&&row.model===old)?.run_id:state.catalog[0].run_id);if(!spec())throw new Error(text('地址中的 SAE 字典不存在。'));state.scope=params.get('scope')||'whole';if(!spec().scopes.includes(state.scope))throw new Error(text('此字典没有对应图像范围的材料。'));state.feature=Number(params.get('feature')||0);state.case=Number(params.get('case')||0);state.mode=params.get('mode')||'spatial';state.panel=params.get('panel')==='findings'?'findings':'explore';el('findings-scope').value=params.get('findings')==='crop'?'crop':'whole';}
window.addEventListener('popstate',()=>run(async()=>{readLocation();el('search').value='';showPanel(state.panel,false,false);await loadSelection({catalog:true,locate:true});if(state.panel==='findings')await loadFindings();}));
window.addEventListener('viewer-language-change',()=>{if(state.catalog.length){controls();featureList();updateCases();showMetadata();drawAll();if(state.panel==='findings')run(loadFindings);}document.title=`SCL-VMI · ${text('热图浏览')}`;});
run(async()=>{const data=await get('/api/heatmaps/catalog');state.catalog=data.dictionaries;if(!location.search&&localStorage.getItem('sclvmi-heatmap-location'))history.replaceState(null,'',localStorage.getItem('sclvmi-heatmap-location'));readLocation();showPanel(state.panel,false,false);await loadSelection({catalog:true,locate:true,resetPoint:true});if(state.panel==='findings')await loadFindings();});
