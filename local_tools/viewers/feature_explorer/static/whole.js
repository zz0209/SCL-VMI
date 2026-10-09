'use strict';
const el = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const state = {model:params.get('model')||'fmcib',feature:Number(params.get('feature')||1976),case:Number(params.get('case')||0),mode:params.get('mode')||'spatial',point:null,meta:null,rects:{},features:[],cases:[],limit:80,request:0,space:false,pan:{},drag:null};
const modes = {spatial:'空间激活',single:'每个 crop 的单点 SAE 值',mean:'每个区域的平均值',maximum:'每个区域的峰值'};
const names = {fmcib:'FMCIB',vista:'VISTA3D'};
let renderer,loadController,drawFrame=0;
const volumeCache=new Map();
const performanceSamples={loads:[],frames:[]};
const format = value => Number(value).toLocaleString(ViewerLanguage.locale,{maximumFractionDigits:3});
const escapeText = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function get(url,signal){const response=await fetch(url,{signal});if(!response.ok){const data=await response.json();throw new Error(data.detail||`HTTP ${response.status}`);}const data=await response.json();if(data.translations)ViewerLanguage.add(data.translations);return data;}
async function binary(url,signal){
  if(volumeCache.has(url)){const data=volumeCache.get(url);volumeCache.delete(url);volumeCache.set(url,data);return data;}
  const response=await fetch(url,{signal});
  if(!response.ok){const data=await response.json();throw new Error(data.detail||`HTTP ${response.status}`);}
  const data=new Float32Array(await response.arrayBuffer());
  volumeCache.set(url,data);
  let bytes=[...volumeCache.values()].reduce((sum,value)=>sum+value.byteLength,0);
  for(const [key,value] of volumeCache){if(bytes<=160*1024*1024)break;volumeCache.delete(key);bytes-=value.byteLength;}
  return data;
}
function loading(message,error=false){
  const shell=document.querySelector('.viewer-shell');
  shell.setAttribute('aria-busy',String(!error));shell.toggleAttribute('data-error',error);
  el('viewer-loading').hidden=false;el('viewer-loading-text').textContent=message;
  document.querySelector('.viewer-grid').style.visibility='hidden';
  state.meta=null;state.rects={};
  for(const axis of [0,1,2]){const canvas=el(`canvas-${axis}`);canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);el(`slice-${axis}`).disabled=true;el(`position-${axis}`).textContent='';}
  el('map-stats').innerHTML='';el('point-value').textContent='';el('map-label').textContent='';el('geometry-note').textContent='';el('feature-evidence').textContent='';
}
function query(extra={}){return new URLSearchParams({model:state.model,case:state.case,feature:state.feature,mode:state.mode,...extra});}
function report(error){if(error.name==='AbortError')return;el('loading').textContent='';loading('影像读取失败，请重新选择或刷新页面。',true);el('error').textContent=error.message;el('error').hidden=false;}
function run(action){Promise.resolve().then(action).catch(report);}
function syncURL(){history.replaceState(null,'',`/whole?${query()}`);}
function activeFeature(){return state.features.find(item=>item.id===state.feature)||{id:state.feature,description:'空间响应待检查'};}
function featureMetric(row,key){
  if(Number.isFinite(row[key]))return {rank:row[key],text:format(row[key])};
  const region=key==='lung_enrichment'?'lung':key==='liver_enrichment'?'liver':null;
  if(region&&row.discovery_region_means?.[region]>0)return {rank:Infinity,text:'参考区域均值为零'};
  return {rank:-Infinity,text:row[key]===null?(region?'两组均无响应':'观察病例无响应'):'统计中'};
}
function featureList(){
  const term=el('search').value.trim().toLowerCase(), key=el('category').value;
  let rows=state.features.filter(row=>`${row.id} ${row.description} ${ViewerLanguage.text(row.description)}`.toLowerCase().includes(term));
  if(key!=='id')rows.sort((a,b)=>{const left=featureMetric(a,key).rank,right=featureMetric(b,key).rank;return left===right?a.id-b.id:right-left;});
  const shown=rows.slice(0,state.limit),selected=rows.find(row=>row.id===state.feature);
  if(selected&&!shown.includes(selected))shown.unshift(selected);
  el('feature-count').textContent=state.features.length.toLocaleString();
  el('feature-list').innerHTML=shown.map(row=>`<button class="feature-item ${row.id===state.feature?'active':''}" data-feature="${row.id}"><strong>${row.id}</strong><span>${escapeText(row.description)}</span><small>${row.id===state.feature?'当前选择':key==='id'?'固定字典':`${escapeText(el('category').selectedOptions[0].text)} ${featureMetric(row,key).text}`}</small></button>`).join('');
  el('more-features').hidden=rows.length<=state.limit;
}
let catalogRequest=0;
async function loadCatalog(selectDefault=false){const id=++catalogRequest,model=state.model;el('feature-list').inert=true;const data=await get(`/api/whole/catalog/${model}`);if(id!==catalogRequest||model!==state.model)return false;state.features=data.features;if(selectDefault&&Number.isInteger(data.preferred_feature))state.feature=data.preferred_feature;el('dictionary-note').textContent=`${names[state.model]} · seed 2025 · ${state.features.length.toLocaleString()} 个 feature。编号对应本模型字典。`;featureList();el('feature-list').inert=false;return true;}
function updateCases(){
  el('case-select').innerHTML=state.cases.map(row=>`<option value="${row.id}" ${!row.ready[state.model]?'disabled':''}>${row.label} · ${row.split}${row.ready[state.model]?'':' · 计算中'}</option>`).join('');
  el('case-select').value=state.case;
  el('case-strip').innerHTML=state.cases.map(row=>`<button class="case-tile ${row.id===state.case?'active':''}" data-case="${row.id}" ${!row.ready[state.model]?'disabled':''}><img class="whole-thumb" src="/api/whole/thumbnail/${row.id}" alt="${row.label} 冠状面 CT" loading="lazy"><strong>${row.label}</strong><span>${row.split}</span><span>${row.ready[state.model]?'完整扫描':'计算中'}</span></button>`).join('');
}
async function updateStatus(){
  const previous=JSON.stringify(state.cases),status=await get('/api/whole/status');state.cases=status.cases;
  const ready=state.cases.reduce((n,row)=>n+Number(row.ready.fmcib)+Number(row.ready.vista),0);
  const p=status.progress;
  el('run-progress').textContent=ready===16?'':p?`${names[p.model]} · CT ${String(p.case).padStart(2,'0')}：${p.windows.toLocaleString()} / ${p.total_windows.toLocaleString()} 个窗口，${p.windows_per_second.toFixed(1)} 窗口/秒；已完成 ${ready}/16 份完整扫描响应。`:'正在准备完整 CT 响应。';
  if(previous!==JSON.stringify(state.cases))updateCases();
}
async function loadMap(resetPoint=false,desiredPoint=null){
  loadController?.abort();loadController=new AbortController();const signal=loadController.signal;
  const id=++state.request,started=performance.now(),selection=query().toString();
  loading(`正在加载 ${names[state.model]} · Feature ${state.feature}…`);el('loading').textContent='读取整张 CT 响应…';el('error').hidden=true;
  el('feature-title').textContent=`${names[state.model]} · Feature ${state.feature}`;el('feature-description').textContent=activeFeature().description;
  el('case-label').textContent=`CT ${String(state.case+1).padStart(2,'0')} · ${state.cases[state.case]?.split||''}`;
  featureList();syncURL();
  if(!state.cases[state.case]?.ready[state.model])throw new Error('当前病例正在计算完整响应，请查看计算进度。');
  const [meta,ct,activity]=await Promise.all([get(`/api/whole/map?${selection}`,signal),binary(`/api/whole/ct-volume/${state.case}`,signal),binary(`/api/whole/activity-volume?${selection}`,signal)]);
  if(id!==state.request)return;
  el('viewer-loading-text').textContent='正在绘制三个方向的影像…';
  await new Promise(requestAnimationFrame);if(id!==state.request)return;
  if(!renderer)renderer=new WholeVolumeRenderer();
  renderer.upload(meta,ct,activity,state.mode);
  state.meta=meta;state.pan={};
  if(resetPoint||!state.point)state.point=meta.shape.map((size,axis)=>Math.floor(size*(axis===2?.65:.5)));
  if(desiredPoint)state.point=[...desiredPoint];
  state.point=state.point.map((value,axis)=>Math.min(meta.shape[axis]-1,Math.max(0,value)));
  el('scale').value=Number(meta.scale.toPrecision(5));
  for(let axis=0;axis<3;axis++){el(`slice-${axis}`).max=meta.shape[axis]-1;el(`slice-${axis}`).value=state.point[axis];}
  const feature=activeFeature();
  el('feature-title').textContent=`${names[state.model]} · Feature ${state.feature}`;
  el('feature-description').textContent=feature.description;
  el('case-label').textContent=`CT ${String(state.case+1).padStart(2,'0')} · ${state.cases[state.case].split}`;
  el('map-label').textContent=`${modes[state.mode]} · ${meta.map_spacing_mm} mm`;
  el('geometry-note').textContent=`${meta.scale_source} · ${state.mode==='spatial'?'最近采样位置显示':'区域内显示同一数值'}`;
  el('map-stats').innerHTML=`<dt>最大值</dt><dd>${format(meta.maximum)}</dd><dt>平均值</dt><dd>${format(meta.mean)}</dd><dt>有响应的位置</dt><dd>${format(meta.nonzero_fraction*100)}%</dd><dt>真实采样网格</dt><dd>${meta.map_shape.join(' × ')}</dd>`;
  el('feature-evidence').textContent=feature.evidence||'器官参考、灰度关系和窗口位置检查正在计算。';
  paintAll();
  document.querySelector('.viewer-grid').style.visibility='visible';el('viewer-loading').hidden=true;document.querySelector('.viewer-shell').setAttribute('aria-busy','false');
  for(const axis of [0,1,2])el(`slice-${axis}`).disabled=false;
  el('loading').textContent='';performanceSamples.loads.push({selection,ms:performance.now()-started});if(performanceSamples.loads.length>100)performanceSamples.loads.shift();
  document.querySelector('.viewer-shell').dataset.loadMs=String(performanceSamples.loads.at(-1).ms);
  document.querySelector('.viewer-shell').dataset.selection=selection;
}
function draw(axis){
  const meta=state.meta;if(!meta||!renderer)return;
  const canvas=el(`canvas-${axis}`),box=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
  if(!box.width||!box.height)return;
  const targetWidth=Math.round(box.width*dpr),targetHeight=Math.round(box.height*dpr);
  if(canvas.width!==targetWidth||canvas.height!==targetHeight){canvas.width=targetWidth;canvas.height=targetHeight;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#0b151a';ctx.fillRect(0,0,box.width,box.height);
  const [u,v]=[0,1,2].filter(value=>value!==axis),nx=meta.shape[u],ny=meta.shape[v];
  const physical=[nx*meta.spacing[u],ny*meta.spacing[v]],zoom=Number(el('zoom').value);
  const unit=Math.min((box.width-32)/physical[0],(box.height-32)/physical[1])*zoom;
  const width=physical[0]*unit,height=physical[1]*unit,pan=state.pan[axis]||[0,0];
  const left=(box.width-width)/2+pan[0],top=(box.height-height)/2+pan[1];
  state.rects[axis]={left,top,width,height,nx,ny,u,v};
  const windows={tissue:[-160,240],lung:[-1000,400],bone:[-500,1500]};
  const scale=Math.max(.001,Number(el('scale').value)),opacity=Number(el('opacity').value)/100,threshold=Number(el('threshold').value)/100,overlay=el('overlay').checked&&!state.space;
  const off=renderer.render(axis,state.point[axis],{window:windows[el('window').value],scale,opacity:overlay?opacity:0,threshold});
  ctx.imageSmoothingEnabled=false;ctx.drawImage(off,left,top,width,height);
  const x=left+(state.point[u]+.5)/nx*width,y=top+(ny-state.point[v]-.5)/ny*height;
  ctx.strokeStyle='rgba(120,223,205,.8)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+height);ctx.moveTo(left,y);ctx.lineTo(left+width,y);ctx.stroke();
  const directions=axis===2?['L','R','A','P']:axis===1?['L','R','S','I']:['P','A','S','I'];
  ctx.fillStyle='#dce8e8';ctx.font='12px "Segoe UI",sans-serif';ctx.textAlign='center';ctx.fillText(directions[0],12,box.height/2);ctx.fillText(directions[1],box.width-12,box.height/2);ctx.fillText(directions[2],box.width/2,15);ctx.fillText(directions[3],box.width/2,box.height-7);
  el(`position-${axis}`).textContent=`${state.point[axis]+1} / ${meta.shape[axis]}`;el(`slice-${axis}`).value=state.point[axis];
}
function paintAll(){
  if(!state.meta)return;
  const started=performance.now();
  [0,1,2].forEach(draw);el('legend-high').textContent=format(el('scale').value);
  const [ct,activity]=renderer.point(state.point);el('point-value').textContent=`${format(ct)} HU · activation ${format(activity)}`;
  performanceSamples.frames.push(performance.now()-started);if(performanceSamples.frames.length>300)performanceSamples.frames.shift();
  document.querySelector('.viewer-shell').dataset.paintMs=String(performanceSamples.frames.at(-1));
}
function drawAll(){if(!drawFrame)drawFrame=requestAnimationFrame(()=>{drawFrame=0;run(paintAll);});}
function movePoint(point){if(!state.meta)return;state.point=point;drawAll();}
function step(axis,delta){if(!state.meta)return;const point=[...state.point];point[axis]=Math.min(state.meta.shape[axis]-1,Math.max(0,point[axis]+delta));run(()=>movePoint(point));}
for(const axis of [0,1,2]){
  const canvas=el(`canvas-${axis}`);
  el(`slice-${axis}`).addEventListener('input',event=>{if(!state.meta)return;const point=[...state.point];point[axis]=Number(event.target.value);movePoint(point);});
  canvas.addEventListener('wheel',event=>{event.preventDefault();step(axis,event.deltaY>0?1:-1);},{passive:false});
  canvas.addEventListener('keydown',event=>{if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();step(axis,['ArrowUp','ArrowRight'].includes(event.key)?1:-1);}});
  canvas.addEventListener('pointerdown',event=>{if(!state.rects[axis])return;canvas.focus();if(event.shiftKey){state.drag={axis,x:event.clientX,y:event.clientY,pan:state.pan[axis]||[0,0]};canvas.setPointerCapture(event.pointerId);return;}const box=canvas.getBoundingClientRect(),r=state.rects[axis],point=[...state.point];point[r.u]=Math.min(r.nx-1,Math.max(0,Math.floor((event.clientX-box.left-r.left)/r.width*r.nx)));point[r.v]=Math.min(r.ny-1,Math.max(0,r.ny-1-Math.floor((event.clientY-box.top-r.top)/r.height*r.ny)));run(()=>movePoint(point));});
  canvas.addEventListener('pointermove',event=>{const drag=state.drag;if(drag?.axis!==axis)return;state.pan[axis]=[drag.pan[0]+event.clientX-drag.x,drag.pan[1]+event.clientY-drag.y];drawAll();});
  canvas.addEventListener('pointerup',()=>state.drag=null);
  canvas.addEventListener('dblclick',()=>{state.pan={};el('zoom').value='1';drawAll();});
}
el('feature-list').addEventListener('click',event=>{const button=event.target.closest('[data-feature]');if(button){state.feature=Number(button.dataset.feature);run(()=>loadMap());}});
el('more-features').onclick=()=>{state.limit+=80;featureList();};
el('search').oninput=()=>{state.limit=80;featureList();};el('category').onchange=()=>{state.limit=80;featureList();};
el('model').onchange=()=>run(async()=>{loadController?.abort();++state.request;loading('正在读取模型目录…');state.model=el('model').value;state.feature=state.model==='fmcib'?1976:0;el('search').value='';if(!await loadCatalog(true))return;updateCases();await loadMap();});
el('mode').onchange=()=>{state.mode=el('mode').value;run(()=>loadMap());};
el('case-select').onchange=()=>{state.case=Number(el('case-select').value);updateCases();run(()=>loadMap(true));};
el('case-strip').onclick=event=>{const button=event.target.closest('[data-case]');if(button&&!button.disabled){state.case=Number(button.dataset.case);updateCases();run(()=>loadMap(true));}};
for(const id of ['overlay','window','zoom','scale'])el(id).onchange=drawAll;
for(const id of ['opacity','threshold'])el(id).oninput=()=>{el(`${id}-value`).textContent=`${el(id).value}%`;drawAll();};
el('peak').onclick=()=>state.meta&&run(()=>movePoint([...state.meta.peak]));
el('center').onclick=()=>state.meta&&run(()=>movePoint(state.meta.shape.map(size=>Math.floor(size/2))));
el('reset-scale').onclick=()=>{if(state.meta){el('scale').value=Number(state.meta.scale.toPrecision(5));drawAll();}};
el('help-toggle').onclick=()=>el('help').hidden=!el('help').hidden;
document.querySelectorAll('.expand').forEach(button=>button.onclick=()=>{const grid=document.querySelector('.viewer-grid'),view=button.closest('.viewport');const expanded=grid.classList.toggle('expanded');document.querySelectorAll('.viewport').forEach(item=>item.classList.toggle('selected-view',expanded&&item===view));button.textContent=expanded?'恢复':'放大';requestAnimationFrame(drawAll);});
document.addEventListener('keydown',event=>{if(event.code==='Space'&&!['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName)){event.preventDefault();state.space=true;drawAll();}});
document.addEventListener('keyup',event=>{if(event.code==='Space'){state.space=false;drawAll();}});
window.addEventListener('blur',()=>{state.space=false;drawAll();});
new ResizeObserver(()=>drawAll()).observe(document.querySelector('.viewer-grid'));
function showPanel(findings){el('explore-panel').hidden=findings;el('findings-panel').hidden=!findings;el('explore-tab').classList.toggle('active',!findings);el('findings-tab').classList.toggle('active',findings);if(!findings)requestAnimationFrame(drawAll);}
el('explore-tab').onclick=()=>showPanel(false);
el('findings-tab').onclick=()=>run(async()=>{showPanel(true);const data=await get('/api/whole/findings');el('findings-list').innerHTML=data.findings.length?data.findings.map(item=>`<article class="finding"><div><h2>${escapeText(item.title)}</h2><p>${escapeText(item.observation)}</p><p class="finding-evidence">${escapeText(item.evidence)}</p><p class="qualification">${escapeText(item.boundary)}</p></div><div class="finding-links">${(item.links||[]).map(link=>`<button data-link='${escapeText(JSON.stringify(link))}'>${names[link.model]} · ${link.feature} · CT ${String(link.case+1).padStart(2,'0')}</button>`).join('')}</div>${item.figure?`<figure><img src="${escapeText(item.figure)}" alt="${escapeText(item.title)}"><figcaption>${escapeText(item.caption||'')}</figcaption></figure>`:''}</article>`).join(''):'<p class="method-note">完整扫描提取与图像检查正在进行。此处将在实际证据完成后显示观察。</p>';});
el('findings-list').onclick=event=>{const button=event.target.closest('[data-link]');if(button)run(async()=>{loadController?.abort();++state.request;loading('正在读取研究观察对应的影像…');const link=JSON.parse(button.dataset.link);Object.assign(state,link);el('model').value=state.model;el('mode').value=state.mode;if(!await loadCatalog())return;updateCases();showPanel(false);await loadMap(true,link.point);});};
run(async()=>{el('model').value=state.model;el('mode').value=state.mode;await updateStatus();await loadCatalog(!params.has('feature'));await loadMap(true);});
setInterval(()=>run(updateStatus),30000);
window.addEventListener('viewer-language-change',()=>{featureList();updateCases();drawAll();});
