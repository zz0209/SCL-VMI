const $ = (id) => document.getElementById(id);
const state = {catalog:null, feature:2029, cases:[], case:null, volume:null, bundle:null, probeVolumes:{}, position:[25,25,25], scale:1, limit:100, featureTicket:0, caseTicket:0, mapTicket:0, space:false, pan:[[0,0],[0,0],[0,0]], layouts:[], interventions:null};
const palette = [[21,11,55],[91,28,110],[170,51,91],[229,107,56],[251,182,26],[252,255,164]];
const planeAxes = [[1,2],[0,2],[0,1]];
const orientationLabels = [['P','A','S','I'],['L','R','S','I'],['L','R','A','P']];
const windows = {lung:[-1000,400],tissue:[-160,240],bone:[-500,1500]};
let controller = new AbortController(), paintPending = false;
const responseCache = new Map();
const planeBuffers = Array.from({length:3},()=>{const canvas=document.createElement('canvas');canvas.width=50;canvas.height=50;const context=canvas.getContext('2d');return {canvas,context,pixels:context.createImageData(50,50)};});
function beginLoading(message) {
  controller.abort();controller=new AbortController();state.ready=false;state.loadStarted=performance.now();
  $('crop-viewer').setAttribute('aria-busy','true');delete $('crop-viewer').dataset.error;
  $('crop-loading').hidden=false;$('crop-loading-message').textContent=message;$('loading').textContent=message;$('error').hidden=true;
  $('map-source').disabled=true;
  for(let axis=0;axis<3;axis++){const canvas=$(`canvas-${axis}`);canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);$(`slice-${axis}`).disabled=true;$(`position-${axis}`).textContent='';}
  for(const id of ['case-label','map-label','point-value','geometry-note'])$(id).textContent='';
}
function finishLoading() {
  state.ready=true;paint();$('crop-viewer').setAttribute('aria-busy','false');$('crop-loading').hidden=true;$('loading').textContent='';
  $('crop-viewer').dataset.loadMs=(performance.now()-state.loadStarted).toFixed(1);
  $('crop-viewer').dataset.selection=`${state.feature}/${state.case}/${activeMap()}`;
  for(let axis=0;axis<3;axis++)$(`slice-${axis}`).disabled=false;
  $('map-source').disabled=false;
}
const fmt = (value, digits=3) => value == null || !Number.isFinite(+value) ? '未定义' : (+value).toFixed(digits);
const pct = (value, digits=1) => value == null ? '未定义' : `${(+value*100).toFixed(digits)}%`;
const clamp = (x,a,b) => Math.min(b,Math.max(a,x));
const featureInfo = () => state.catalog.features[state.feature];
const interpretation = (id) => state.catalog.interpretations[id] || {name:'待查看的空间响应',group:'尚未命名',description:'此 feature 的全部病例热图与数值检查可以浏览。当前没有经过图像检查的内容描述。'};
function problem(error) {
  if(error.name==='AbortError')return;
  state.ready=false;$('crop-viewer').dataset.error='true';$('crop-viewer').setAttribute('aria-busy','false');
  $('crop-loading').hidden=false;$('crop-loading-message').textContent='影像读取失败，请重新选择或刷新页面。';
  $('error').textContent=`读取或显示失败：${error.message}。请刷新页面重新读取；服务记录保留原始错误。`;
  $('error').hidden=false;$('loading').textContent='读取失败';$('map-source').disabled=false;
  $('intervention-status').dataset.loading='false';if(!state.interventions)$('intervention-status').textContent='读取失败';console.error(error);
}
async function api(url,binary=false) { const key=`${binary}/${url}`;if(responseCache.has(key)){const value=responseCache.get(key);responseCache.delete(key);responseCache.set(key,value);return value;}const signal=controller.signal;const response=await fetch(url,{signal,priority:'high'});if(!response.ok)throw new Error(`${response.status} ${await response.text()}`);const value=binary?new Float32Array(await response.arrayBuffer()):await response.json();signal.throwIfAborted();responseCache.set(key,value);while(responseCache.size>96)responseCache.delete(responseCache.keys().next().value);return value; }
function guarded(fn) { return (...args) => Promise.resolve(fn(...args)).catch(problem); }
function element(tag, text, className) { const node=document.createElement(tag); if(text!=null)node.textContent=text; if(className)node.className=className; return node; }
function transformPoint(matrix, point) { return matrix.slice(0,3).map(row=>row[3]+row.slice(0,3).reduce((sum,value,i)=>sum+value*point[i],0)); }
function index3(point, shape=[50,50,50]) { return (point[0]*shape[1]+point[1])*shape[2]+point[2]; }
function inInterior(point) { return state.volume.interior_bounds[0].every((lower,i)=>point[i]>=lower && point[i]<=state.volume.interior_bounds[1][i]); }
function activeMap() { return $('map-source').value; }
const probeModes=['blur_1mm','hu_plus100','hu_minus100'];
function caseValue(row) { const value=row[$('case-sort').value];return $('case-sort').value==='effect'?`${fmt(value*100,3)} pp`:fmt(value); }
function mapValue(point) {
  if($('scope').value==='interior' && !inInterior(point))return 0;
  if($('sampling').value==='native'){
    const native=transformPoint(state.bundle.canonical_to_native,point).map(value=>clamp(Math.round(value),0,12));
    return state.bundle.native[activeMap()][index3(native,[13,13,13])];
  }
  return state.bundle.maps[activeMap()][index3(point)];
}
function color(value, signed) {
  if(signed && value<0) {const t=clamp(-value/state.scale,0,1); return [30+20*t,110+77*t,130+80*t];}
  const t=clamp(value/state.scale,0,1)*(palette.length-1), i=Math.min(palette.length-2,Math.floor(t)), f=t-i;
  return palette[i].map((v,j)=>Math.round(v*(1-f)+palette[i+1][j]*f));
}
function renderFeatures() {
  if(!state.catalog)return;
  const search=$('search').value.trim().toLowerCase(), category=$('category').value;
  let rows=state.catalog.features.filter(row=>{
    const item=interpretation(row.feature_id);
    const matches=category==='all'||(category==='reviewed' && state.catalog.interpretations[row.feature_id])||item.group===category;
    return matches && `${row.feature_id} ${item.name} ${item.group} ${CropI18n.text(item.name)} ${CropI18n.text(item.group)}`.toLowerCase().includes(search);
  });
  const sort=$('sort').value;
  const metric={r2:'confirmation_combined_r2',entropy:'confirmation_entropy',repeatability:'patient_mean_minimum',position:'confirmation_position_variance_fraction'}[sort];
  rows.sort((a,b)=>sort==='id'?a.feature_id-b.feature_id:((sort==='entropy'?1:-1)*((a[metric]??-999)-(b[metric]??-999))||a.feature_id-b.feature_id));
  $('feature-count').textContent=`${rows.length}`;
  $('feature-list').replaceChildren(...rows.slice(0,state.limit).map(row=>{
    const item=interpretation(row.feature_id), button=element('button',null,`feature-item${row.feature_id===state.feature?' active':''}`);
    button.setAttribute('aria-pressed',String(row.feature_id===state.feature));
    button.append(element('strong',String(row.feature_id)),element('span',item.name),element('small',item.group));
    button.addEventListener('click',guarded(()=>selectFeature(row.feature_id)));
    return button;
  }));
  $('more-features').hidden=rows.length<=state.limit;
}
function fact(label,value) {const span=element('span',label);span.append(element('strong',value));return span;}
function bar(label,value,maximum=1,control=false,format=fmt) {
  const row=element('div',null,`bar-row${control?' control':''}`), track=element('div',null,'bar-track'), fill=element('div',null,'bar-fill');
  fill.style.width=`${clamp(value/maximum*100,0,100)}%`;track.append(fill);
  row.append(element('span',label),track,element('span',format(value),'bar-value'));return row;
}
function renderProfile() {
  const row=featureInfo(), item=interpretation(state.feature);
  $('feature-title').textContent=`Feature ${state.feature} · ${item.name}`;
  $('feature-description').textContent=item.description;
  $('feature-facts').replaceChildren(fact('观察类别',item.group),fact('内部有响应患者',`${row.confirmation_active_patients} / 64`),fact('跨 seed 最低均值',fmt(row.patient_mean_minimum)),fact('平移后相关性',fmt(row.confirmation_translation_correlation)));
  $('regression-chart').replaceChildren(bar('位置变量',row.confirmation_position_r2),bar('图像内容变量',row.confirmation_content_r2),bar('内容与位置',row.confirmation_combined_r2));
  const metrics=[['响应加权灰度',`${fmt(row.confirmation_weighted_hu,0)} HU`],['响应加权梯度',`${fmt(row.confirmation_weighted_gradient,1)} HU / 输入体素`],['空间熵（0–1）',fmt(row.confirmation_entropy)],['内部激活位置比例',pct(row.confirmation_active_fraction)],['外部两层响应占比',pct(row.confirmation_outer_mass_fraction)],['跨 seed 相关性下限',fmt(row.patient_confirmation_lower95_minimum)]];
  $('spatial-stats').replaceChildren(...metrics.flatMap(([a,b])=>[element('dt',a),element('dd',b)]));
  const probe=state.catalog.probes.find(value=>value.feature_id===state.feature),labels=['1 mm 模糊','增加 100 HU','减少 100 HU'];
  const maximum=Math.max(1,...probeModes.map(mode=>probe[`${mode}_ratio`]??0))*1.1;
  $('input-chart').replaceChildren(bar('原始输入',1,maximum,true,value=>`${fmt(value,2)}×`),...probeModes.map((mode,i)=>bar(labels[i],probe[`${mode}_ratio`],maximum,false,value=>`${fmt(value,2)}×`)));
  $('input-stats').replaceChildren(...probeModes.flatMap((mode,i)=>[element('dt',`${labels[i]}后的相关性`),element('dd',`${fmt(probe[`${mode}_correlation`])} · ${probe[`${mode}_coactive_patients`]} 人`)]));
}
async function selectFeature(feature,desiredCase=null) {
  beginLoading('读取 feature 与病例…');
  const ticket=++state.featureTicket;
  ++state.caseTicket;
  state.feature=feature;state.bundle=null;state.interventions=null;state.cases=[];$('case-strip').replaceChildren();$('case-select').replaceChildren();renderInterventions();
  ++state.mapTicket;
  const reviewed=Boolean(state.catalog.interpretations[feature]);
  $('case-sort').querySelector('option[value="effect"]').disabled=!reviewed;
  for(const mode of probeModes)$('map-source').querySelector(`option[value="${mode}"]`).disabled=!reviewed;
  if(!reviewed&&$('case-sort').value==='effect')$('case-sort').value='maximum';
  $('loading').textContent='读取 feature 与病例…';$('error').hidden=true;
  renderFeatures();renderProfile();
  state.scale=Math.max(.01,state.catalog.scales[feature]);$('scale').value=state.scale.toFixed(3);
  const rows=await api(`/api/cases/${feature}?split=${$('split').value}&order=${$('case-sort').value}`);
  if(ticket!==state.featureTicket)return;
  state.cases=rows;renderCases();
  if(desiredCase!==null&&!rows.some(row=>row.case_index===desiredCase))throw new Error('The requested case is outside the selected sample');
  await selectCase(desiredCase!==null?desiredCase:rows[0].case_index);
}
function renderCases() {
  const rows=state.cases, featured=[...new Set([...rows.slice(0,6).map((_,i)=>i),Math.floor(rows.length/2),rows.length-1])];
  $('case-strip').replaceChildren(...featured.map(i=>{
    const row=rows[i], button=element('button',null,`case-tile${row.case_index===state.case?' active':''}`);
    button.dataset.case=row.case_index;button.setAttribute('aria-label',`查看 ${row.label}，排名 ${row.rank}`);
    const wrapper=element('span',null,'thumbnail-wrap'),spinner=element('span',null,'loading-spinner'),img=element('img');
    wrapper.setAttribute('aria-busy','true');spinner.setAttribute('aria-hidden','true');wrapper.append(spinner,img);
    img.alt=`${row.label} 的峰值所在 CT 横断面`;img.decoding='async';img.fetchPriority='low';
    img.addEventListener('load',()=>{wrapper.dataset.ready='true';wrapper.setAttribute('aria-busy','false');});
    img.addEventListener('error',()=>{wrapper.dataset.error='true';wrapper.setAttribute('aria-busy','false');wrapper.replaceChildren(element('span','缩略图读取失败'));});
    img.src=`/api/thumbnail/${state.feature}/${row.case_index}`;
    const value=row[$('case-sort').value];
    button.append(wrapper,element('strong',`${row.label} · 第 ${row.rank} 位`),element('span',`${$('case-sort').value==='effect'?'概率影响':'响应'} ${caseValue(row)}`));
    button.addEventListener('click',guarded(()=>selectCase(row.case_index)));return button;
  }));
  $('case-select').replaceChildren(...rows.map(row=>{const option=element('option',`第 ${row.rank} 位 · ${row.label} · ${caseValue(row)}`);option.value=row.case_index;return option;}));
}
async function selectCase(caseIndex) {
  beginLoading('读取 CT 和空间响应…');
  const ticket=++state.caseTicket, feature=state.feature;
  state.case=caseIndex;state.bundle=null;state.interventions=null;renderInterventions();
  state.probeVolumes={};++state.mapTicket;
  $('loading').textContent='读取 CT 和空间响应…';$('error').hidden=true;
  document.querySelectorAll('.case-tile').forEach(button=>button.classList.toggle('active',+button.dataset.case===caseIndex));
  $('case-select').value=String(caseIndex);
  const mode=activeMap();
  const [metadata,ct,map,interventions]=await Promise.all([api(`/api/crop/metadata/${feature}/${caseIndex}`),api(`/api/crop/volume/${caseIndex}`,true),api(`/api/crop/map/${feature}/${caseIndex}/${mode}`,true),api(`/api/interventions/${feature}?case=${caseIndex}`)]);
  if(ticket!==state.caseTicket || feature!==state.feature)return;
  const volume={...metadata,volume:ct};state.volume=volume;state.bundle={...metadata,maps:{},native:{}};installMap(mode,map);state.pan=[[0,0],[0,0],[0,0]];
  $('zoom').value='1';
  if(ticket!==state.caseTicket || feature!==state.feature)return;
  locatePeak();finishLoading();
  const row=state.cases.find(value=>value.case_index===caseIndex);
  $('case-label').textContent=`${volume.label} · 第 ${row.rank} / ${state.cases.length} 位`;
  $('rank-note').textContent=`${volume.split} · 当前排序值 ${caseValue(row)}`;
  $('previous-case').disabled=row.rank===1;$('next-case').disabled=row.rank===state.cases.length;
  $('loading').textContent='';
  $('geometry-note').textContent=`CT 50³ · activation 13³ · RAS 方向 · 采样间距 ${volume.spacing.map(v=>fmt(v,2)).join(' / ')} mm`;
  updateURL();
  if(ticket!==state.caseTicket || feature!==state.feature)return;
  state.interventions=interventions;renderInterventions();
}
function updateURL() { const parameters=new URLSearchParams({feature:state.feature,case:state.case,split:$('split').value});history.replaceState(null,'',`/?${parameters}`); }
async function reorderCases() {
  beginLoading('读取病例排序…');state.bundle=null;++state.caseTicket;++state.mapTicket;$('case-strip').replaceChildren();
  const ticket=++state.featureTicket, feature=state.feature;
  $('loading').textContent='读取病例排序…';
  const rows=await api(`/api/cases/${feature}?split=${$('split').value}&order=${$('case-sort').value}`);
  if(ticket!==state.featureTicket)return;
  state.cases=rows;renderCases();await selectCase(rows[0].case_index);
}
function locatePeak() {
  if(!state.bundle || !state.bundle.native[activeMap()])return;
  const native=state.bundle.native[activeMap()], interior=$('scope').value==='interior', signed=activeMap()==='residual';
  let best=-1, point=[6,6,6];
  for(let x=0;x<13;x++)for(let y=0;y<13;y++)for(let z=0;z<13;z++){
    if(interior && [x,y,z].some(v=>v<2||v>10))continue;
    const value=native[index3([x,y,z],[13,13,13])], strength=signed?Math.abs(value):value;
    if(strength>best){best=strength;point=[x,y,z];}
  }
  state.position=best>0?transformPoint(state.bundle.native_to_canonical,point).map(v=>clamp(Math.round(v),0,49)):[25,25,25];
  draw();
}
function draw() {
  if(paintPending||!state.ready)return;paintPending=true;requestAnimationFrame(()=>{paintPending=false;paint();});
}
function paint() {
  if(!state.ready||!state.volume || !state.bundle || !state.bundle.maps[activeMap()])return;
  const started=performance.now();
  for(let axis=0;axis<3;axis++)drawPlane(axis);
  const source=activeMap(), names={raw:`Feature ${state.feature} · seed 2025`,template:'discovery 平均位置模板',residual:'当前响应减去位置模板',seed2026:`Feature ${state.bundle.counterparts['2026']} · seed 2026`,seed2027:`Feature ${state.bundle.counterparts['2027']} · seed 2027`,blur_1mm:'1 mm 模糊后的 CT 与响应',hu_plus100:'增加 100 HU 后的 CT 与响应',hu_minus100:'减少 100 HU 后的 CT 与响应'};
  $('map-label').textContent=names[source];
  const index=index3(state.position), hu=(probeModes.includes(source)?state.probeVolumes[source]:state.volume.volume)[index], activation=mapValue(state.position);
  $('point-value').textContent=`CT ${hu} HU · activation ${fmt(activation)} · [${state.position.join(', ')}]`;
  $('colorbar').classList.toggle('signed',source==='residual');
  $('legend-low').textContent=source==='residual'?fmt(-state.scale):'0';$('legend-high').textContent=fmt(state.scale);
  $('opacity-value').textContent=`${$('opacity').value}%`;$('threshold-value').textContent=`${$('threshold').value}%`;
  $('crop-viewer').dataset.paintMs=(performance.now()-started).toFixed(2);
}
function drawPlane(axis) {
  const canvas=$(`canvas-${axis}`), rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;
  const ratio=window.devicePixelRatio||1,newWidth=Math.round(rect.width*ratio),newHeight=Math.round(rect.height*ratio);if(canvas.width!==newWidth)canvas.width=newWidth;if(canvas.height!==newHeight)canvas.height=newHeight;
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.fillStyle='#0b151a';ctx.fillRect(0,0,rect.width,rect.height);
  const [horizontal,vertical]=planeAxes[axis], spacing=state.volume.spacing;
  const fit=Math.min((rect.width-38)/(50*spacing[horizontal]),(rect.height-30)/(50*spacing[vertical]))*+$('zoom').value;
  const width=50*spacing[horizontal]*fit,height=50*spacing[vertical]*fit;
  const left=(rect.width-width)/2+state.pan[axis][0],top=(rect.height-height)/2+state.pan[axis][1];
  state.layouts[axis]={left,top,width,height};
  const {canvas:offscreen,context:sub,pixels}=planeBuffers[axis], windowRange=windows[$('window').value], signed=activeMap()==='residual';
  const source=activeMap(),ct=probeModes.includes(source)?state.probeVolumes[source]:state.volume.volume,map=state.bundle.maps[source],sampling=$('sampling').value,interior=$('scope').value==='interior',bounds=state.volume.interior_bounds,matrix=state.bundle.canonical_to_native,native=state.bundle.native[source];
  const opacity=+$('opacity').value/100, threshold=+$('threshold').value/100, overlay=$('overlay').checked&&!state.space;
  for(let row=0;row<50;row++)for(let column=0;column<50;column++){
    const point=state.position.slice();point[horizontal]=column;point[vertical]=49-row;
    const index=index3(point), gray=clamp((ct[index]-windowRange[0])/(windowRange[1]-windowRange[0]),0,1)*255;
    let value=map[index];
    if(sampling==='native'){const nx=clamp(Math.round(matrix[0][3]+matrix[0][0]*point[0]+matrix[0][1]*point[1]+matrix[0][2]*point[2]),0,12),ny=clamp(Math.round(matrix[1][3]+matrix[1][0]*point[0]+matrix[1][1]*point[1]+matrix[1][2]*point[2]),0,12),nz=clamp(Math.round(matrix[2][3]+matrix[2][0]*point[0]+matrix[2][1]*point[1]+matrix[2][2]*point[2]),0,12);value=native[(nx*13+ny)*13+nz];}
    if(interior&&(point[0]<bounds[0][0]||point[0]>bounds[1][0]||point[1]<bounds[0][1]||point[1]>bounds[1][1]||point[2]<bounds[0][2]||point[2]>bounds[1][2]))value=0;
    const strength=Math.abs(value)/state.scale;
    const alpha=overlay&&strength>=threshold&&value!==0?opacity*Math.sqrt(Math.min(strength,1)):0;
    const rgb=alpha>0?color(value,signed):[gray,gray,gray], offset=(row*50+column)*4;
    for(let c=0;c<3;c++)pixels.data[offset+c]=gray*(1-alpha)+rgb[c]*alpha;
    pixels.data[offset+3]=255;
  }
  sub.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=$('sampling').value==='linear';ctx.drawImage(offscreen,left,top,width,height);
  if($('crosshair').checked){
    const x=left+(state.position[horizontal]+.5)/50*width,y=top+(49-state.position[vertical]+.5)/50*height;
    ctx.strokeStyle='#59d0ce';ctx.lineWidth=.8;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(x-5,y);ctx.moveTo(x+5,y);ctx.lineTo(left+width,y);ctx.moveTo(x,top);ctx.lineTo(x,y-5);ctx.moveTo(x,y+5);ctx.lineTo(x,top+height);ctx.stroke();ctx.setLineDash([]);
  }
  const labels=orientationLabels[axis];ctx.font='11px "Segoe UI",sans-serif';ctx.fillStyle='#afc6cf';ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.fillText(labels[0],10,rect.height/2);ctx.fillText(labels[1],rect.width-10,rect.height/2);ctx.fillText(labels[2],rect.width/2,9);ctx.fillText(labels[3],rect.width/2,rect.height-8);
  const barLength=10*fit;ctx.strokeStyle='#adbec2';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(17,rect.height-15);ctx.lineTo(17+barLength,rect.height-15);ctx.stroke();ctx.textAlign='left';ctx.font='10px "Segoe UI",sans-serif';ctx.fillText('10 mm',17,rect.height-25);
  $(`slice-${axis}`).value=state.position[axis];$(`position-${axis}`).textContent=`${state.position[axis]+1} / 50`;
}
function svgNode(tag,attributes,text) {const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value]of Object.entries(attributes))node.setAttribute(key,String(value));if(text!=null)node.textContent=text;return node;}
function renderInterventions() {
  const data=state.interventions;
  $('intervention-status').dataset.loading=String(!data);
  if(!data){$('intervention-status').textContent='正在读取干预结果…';$('dose-chart').replaceChildren();$('control-chart').replaceChildren();$('case-effect').textContent='';return;}
  if(data.status!=='completed'){$('intervention-status').textContent=data.status==='not_studied'?'此 feature 尚未运行干预实验。':'干预实验正在运行。';$('dose-chart').replaceChildren();$('control-chart').replaceChildren();$('case-effect').textContent='';return;}
  $('intervention-status').textContent='';
  const sample=data.case[0];
  const svg=$('dose-chart');svg.replaceChildren();
  if(sample){
    const values=[sample.original,sample.ablate_025,sample.ablate_050,sample.ablate_075,sample.ablate_100];
    const low=Math.max(0,Math.min(...values)-.006),high=Math.min(1,Math.max(...values)+.006),x=i=>55+i*115,y=v=>190-(v-low)/(high-low)*165;
    for(let i=0;i<4;i++){
      const value=low+(high-low)*i/3,yy=y(value);
      svg.append(svgNode('line',{x1:55,x2:520,y1:yy,y2:yy,stroke:'#e0e7e2'}),svgNode('text',{x:46,y:yy+4,'text-anchor':'end',fill:'#596d65','font-size':11},pct(value,1)));
    }
    svg.append(svgNode('polyline',{points:values.map((v,i)=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:'#176c59','stroke-width':2}));
    values.forEach((value,i)=>{svg.append(svgNode('circle',{cx:x(i),cy:y(value),r:4,fill:'#176c59'}),svgNode('text',{x:x(i),y:210,'text-anchor':'middle',fill:'#596d65','font-size':11},`${i*25}%`));});
    svg.append(svgNode('text',{x:287,y:229,'text-anchor':'middle',fill:'#596d65','font-size':11},'减弱比例 t'));
    const difference=(sample.ablate_100-sample.original)*100;
    $('case-effect').textContent=`分类器概率 ${pct(sample.original,2)} → ${pct(sample.ablate_100,2)}；变化 ${difference>=0?'+':''}${fmt(difference,3)} 个百分点。表示改变量 / 原表示范数：${pct(sample.relative_delta_norm,2)}。`;
  }
  const summary=data.summary.find(row=>row.feature_split==='confirmation'), keys=['ablate_100','spatial_shuffle','random_direction','interior_only'];
  const labels=['原方向删除','空间打乱控制','随机方向控制','仅删除内部响应'];
  const maximum=Math.max(...keys.map(key=>summary[`${key}_mean_absolute_delta`]),.00001)*1.05;
  $('control-chart').replaceChildren(...keys.map((key,i)=>bar(labels[i],summary[`${key}_mean_absolute_delta`],maximum,i===1||i===2,value=>`${fmt(value*100,3)} pp`)));
}
function showPanel(name) { const explorer=name==='explore';$('explore-panel').hidden=!explorer;$('findings-panel').hidden=explorer;$('explore-tab').classList.toggle('active',explorer);$('findings-tab').classList.toggle('active',!explorer);if(explorer)requestAnimationFrame(draw); }
function renderFindings() {
  const findings=state.catalog.findings;
  $('findings-list').replaceChildren(...findings.items.map(item=>{
    const article=element('article',null,'finding'),body=element('div'), links=element('div',null,'finding-links');
    body.append(element('h2',item.title),element('p',item.observation),element('p',item.evidence,'finding-evidence'),element('p',item.boundary,'qualification'));
    for(const id of item.features){const button=element('button',`Feature ${id}`);button.addEventListener('click',guarded(async()=>{showPanel('explore');$('category').value=state.catalog.interpretations[id]?'reviewed':'all';await selectFeature(id);window.scrollTo({top:0,behavior:'smooth'});}));links.append(button);}
    article.append(body,links);return article;
  }));
  if(!findings.items.length)$('findings-list').append(element('p','图像检查与实验分析正在进行。'));
  $('source-list').replaceChildren(...findings.sources.map(source=>{const row=element('div',null,'source-row');row.append(element('strong',source.name),element('span',source.method));return row;}));
}
for(let axis=0;axis<3;axis++){
  const canvas=$(`canvas-${axis}`);let drag=null;
  $(`slice-${axis}`).addEventListener('input',event=>{state.position[axis]=+event.target.value;draw();});
  canvas.addEventListener('wheel',event=>{if(!state.bundle)return;event.preventDefault();state.position[axis]=clamp(state.position[axis]+(event.deltaY>0?1:-1),0,49);draw();},{passive:false});
  canvas.addEventListener('keydown',event=>{if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();state.position[axis]=clamp(state.position[axis]+(['ArrowUp','ArrowRight'].includes(event.key)?1:-1),0,49);draw();}});
  const positionAt=event=>{if(!state.bundle)return;const rect=canvas.getBoundingClientRect(),layout=state.layouts[axis],x=(event.clientX-rect.left-layout.left)/layout.width,y=(event.clientY-rect.top-layout.top)/layout.height;if(x<0||x>1||y<0||y>1)return;const [h,v]=planeAxes[axis];state.position[h]=clamp(Math.floor(x*50),0,49);state.position[v]=clamp(49-Math.floor(y*50),0,49);draw();};
  canvas.addEventListener('pointerdown',event=>{canvas.focus();canvas.setPointerCapture(event.pointerId);drag={x:event.clientX,y:event.clientY,pan:state.pan[axis].slice(),shift:event.shiftKey};if(!event.shiftKey)positionAt(event);});
  canvas.addEventListener('pointermove',event=>{if(!drag)return;if(drag.shift){state.pan[axis]=[drag.pan[0]+event.clientX-drag.x,drag.pan[1]+event.clientY-drag.y];draw();}else positionAt(event);});
  canvas.addEventListener('pointerup',()=>{drag=null;});canvas.addEventListener('pointercancel',()=>{drag=null;});
  canvas.addEventListener('dblclick',()=>{state.pan[axis]=[0,0];$('zoom').value='1';draw();});
}
for(const id of ['search','category','sort'])$(id).addEventListener(id==='search'?'input':'change',()=>{state.limit=100;renderFeatures();});
$('more-features').addEventListener('click',()=>{state.limit+=100;renderFeatures();});
for(const id of ['overlay','crosshair','opacity','threshold','window','zoom','sampling','scope'])$(id).addEventListener('input',draw);
async function loadMapSource(reposition=true) {
  const ticket=++state.mapTicket,feature=state.feature,caseIndex=state.case,mode=activeMap();
  if(!state.bundle){await selectCase(caseIndex);return;}
  beginLoading('读取 CT 和空间响应…');
  if(!state.bundle.maps[mode]){
    const data=await api(`/api/crop/map/${feature}/${caseIndex}/${mode}`,true);
    if(ticket!==state.mapTicket||feature!==state.feature||caseIndex!==state.case)return;
    installMap(mode,data);
    $('loading').textContent='';
  }
  if(reposition)locatePeak();finishLoading();renderCaseStatus();
}
function installMap(mode,data) {const expected=125000+2197+(probeModes.includes(mode)?125000:0);if(data.length!==expected)throw new Error('Invalid heatmap array length');state.bundle.maps[mode]=data.subarray(0,125000);state.bundle.native[mode]=data.subarray(125000,127197);if(probeModes.includes(mode))state.probeVolumes[mode]=data.subarray(127197);}
function renderCaseStatus(){const row=state.cases.find(value=>value.case_index===state.case);$('case-label').textContent=`${state.volume.label} · 第 ${row.rank} / ${state.cases.length} 位`;$('geometry-note').textContent=`CT 50³ · activation 13³ · RAS 方向 · 采样间距 ${state.volume.spacing.map(v=>fmt(v,2)).join(' / ')} mm`;}
$('map-source').addEventListener('change',guarded(()=>loadMapSource(false)));
$('scale').addEventListener('input',()=>{const value=+$('scale').value;if(value>0){state.scale=value;draw();}});
$('reset-scale').addEventListener('click',()=>{state.scale=Math.max(.01,state.catalog.scales[state.feature]);$('scale').value=state.scale.toFixed(3);draw();});
$('peak').addEventListener('click',locatePeak);$('center').addEventListener('click',()=>{state.position=[25,25,25];state.pan=[[0,0],[0,0],[0,0]];draw();});
$('help-toggle').addEventListener('click',()=>{$('help').hidden=!$('help').hidden;});
$('explore-tab').addEventListener('click',()=>showPanel('explore'));$('findings-tab').addEventListener('click',()=>showPanel('findings'));
for(const id of ['split','case-sort'])$(id).addEventListener('change',guarded(reorderCases));
$('case-select').addEventListener('change',guarded(event=>selectCase(+event.target.value)));
$('middle-case').addEventListener('click',guarded(()=>selectCase(state.cases[Math.floor(state.cases.length/2)].case_index)));
$('low-case').addEventListener('click',guarded(()=>selectCase(state.cases[state.cases.length-1].case_index)));
for(const [id,offset]of [['previous-case',-1],['next-case',1]])$(id).addEventListener('click',guarded(()=>{const position=state.cases.findIndex(row=>row.case_index===state.case);return selectCase(state.cases[clamp(position+offset,0,state.cases.length-1)].case_index);}));
for(const name of ['profile','intervention','input'])$(`${name}-tab`).addEventListener('click',()=>{for(const other of ['profile','intervention','input']){$(`${other}-panel`).hidden=other!==name;$(`${other}-tab`).classList.toggle('active',other===name);}if(name==='intervention')renderInterventions();});
document.querySelectorAll('.expand').forEach(button=>button.addEventListener('click',()=>{const viewport=button.closest('.viewport'),grid=viewport.parentElement,expanded=grid.classList.toggle('expanded');document.querySelectorAll('.viewport').forEach(node=>node.classList.toggle('selected-view',node===viewport&&expanded));document.querySelectorAll('.expand').forEach(node=>node.textContent='放大');button.textContent=expanded?'恢复三方向':'放大';requestAnimationFrame(draw);}));
window.addEventListener('resize',draw);
window.addEventListener('keydown',event=>{if(event.code==='Space'&&!['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement.tagName)){event.preventDefault();state.space=true;draw();}});
window.addEventListener('keyup',event=>{if(event.code==='Space'){state.space=false;draw();}});
window.addEventListener('blur',()=>{state.space=false;draw();});
guarded(async()=>{
  state.catalog=await api('/api/catalog');
  CropI18n.add(state.catalog.translations);
  const parameters=new URLSearchParams(location.search), feature=Number(parameters.get('feature')??2029), caseIndex=parameters.has('case')?Number(parameters.get('case')):null;
  if(parameters.has('split')&&['confirmation','discovery','all'].includes(parameters.get('split')))$('split').value=parameters.get('split');
  if(!state.catalog.features.some(row=>row.feature_id===feature))throw new Error('URL 中的 feature 编号无效');
  if(!state.catalog.interpretations[feature])$('category').value='all';
  renderFindings();$('run-note').textContent='20261009_feature_exploration';
  await selectFeature(feature,caseIndex);
})();
