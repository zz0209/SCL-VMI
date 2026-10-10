'use strict';
const $ = id => document.getElementById(id);
const words = {
  library:['SAE 字典','SAE dictionaries'], crop:['结节研究','Nodule research'], whole:['完整 CT','Whole CT'],
  title:['查看已验证的 SAE 字典','Explore validated SAE dictionaries'], intro:['选择模型层与随机种子，在实际 CT 输入中查看 feature 响应。','Choose a model layer and seed to inspect feature responses in actual CT inputs.'],
  emptyTitle:['字典正在接受验证','Dictionary validation is in progress'], emptyText:['完成训练、分类器检查与重复性分析后，选定字典会出现在这里。','Selected dictionaries appear after training, classifier checks and repeatability analysis.'], refresh:['重新读取','Reload'],
  dictionary:['模型层与种子','Model layer and seed'], feature:['Feature 编号','Feature ID'], previous:['上一个','Previous'], next:['下一个','Next'], case:['病例','Case'], strongest:['最强响应病例','Strongest response case'],
  window:['CT 显示窗','CT window'], lung:['肺窗','Lung'], tissue:['组织窗','Tissue'], bone:['骨窗','Bone'], overlay:['显示热图','Show heatmap'], opacity:['透明度','Opacity'], scale:['颜色上限','Color maximum'], peak:['定位峰值','Locate peak'], center:['回到中心','Center'],
  axial:['横断面','Axial'], coronal:['冠状面','Coronal'], sagittal:['矢状面','Sagittal'], activation:['激活强度','Activation'], evidence:['字典验证结果与使用范围','Dictionary checks and scope'],
  navigation:['影像查看器','Image viewers'], pooledNote:['全局字典对整个输入给出一个响应值。其分类器检查使用重建的通道平均值，并保留原始空间残差。','A global dictionary gives one response for the entire input. Its classifier check uses reconstructed channel means and retains the original spatial residual.'],
  scaleNote:['默认颜色上限使用该 feature 在这 128 名患者中的最大响应，并在病例切换中保持一致。空间热图使用插值显示；网格大小标明模型原始分辨率。','The default color maximum uses this feature’s largest response across the 128 displayed patients and stays fixed across cases. Spatial heatmaps use interpolation; grid dimensions show the original model resolution.'],
  boundary:['这些字典通过表示重建与固定分类器检查。单个 feature 的医学含义需要图像证据；不同种子的相同编号代表各自字典中的 feature。这里的 128 名患者来自开发集，独立测试集保留。','These dictionaries passed representation reconstruction and fixed-classifier checks. Medical meaning requires image evidence. Identical feature IDs in different seeds refer to their own dictionaries. The 128 patients shown come from development; the independent test partition remains reserved.'],
  loading:['正在读取影像与响应…','Loading images and responses…'], loadCatalog:['正在读取字典…','Loading dictionaries…'], error:['读取失败。请检查本地服务，然后重新读取页面。','Loading failed. Check the local service, then reload this page.'],
  features:['特征数量','Features'], activity:['目标平均激活','Target mean activity'], patient:['开发集病例','Development case'], grid:['原始网格','Native grid'], interpolation:['插值显示','Interpolated display'], pooled:['整个 crop 的响应','Whole-crop response'], maximum:['最大响应','Maximum response'], mean:['平均响应','Mean response'], fraction:['激活位置','Active positions'],
  fvu:['开发集 FVU','Development FVU'], cosine:['平均 cosine','Mean cosine'], mae:['分类概率 MAE','Probability MAE'], auc:['重建后 AUROC','Reconstructed AUROC'], inactive:['未激活特征比例','Inactive feature fraction'], hash:['字典 SHA-256','Dictionary SHA-256'], inactiveMap:['当前病例中该 feature 没有激活。','This feature is inactive in the current case.'],
};
const t = key => words[key][ViewerLanguage.value === 'en' ? 1 : 0];
const state = {catalog:null, spec:null, meta:null, volume:null, map:null, position:[0,0,0], request:null, scaleKey:null, rectangles:{}, loadingKey:null, error:null};
const url = new URL(location.href);
function language() {
  document.querySelectorAll('[data-key]').forEach(node => {node.textContent=t(node.dataset.key);});
  for (const [axis,key] of [[0,'sagittal'],[1,'coronal'],[2,'axial']]) {
    $(`view-${axis}`).setAttribute('aria-label',t(key));
    $(`slice-${axis}`).setAttribute('aria-label',t(key));
  }
  document.title = `SCL-VMI · ${t('library')}`;
  document.querySelector('nav').setAttribute('aria-label',t('navigation'));
  if(state.loadingKey) $('loading').textContent=t(state.loadingKey);
  if(state.error) $('error').textContent=`${t('error')} ${state.error}`;
  if (state.meta) showFacts();
}
function node(tag, text) {const element=document.createElement(tag); element.textContent=text; return element;}
function factsLine(label, value) {const span=node('span', `${label}: `); span.append(node('strong', value)); return span;}
function showFacts() {
  const spec=state.spec, meta=state.meta;
  $('facts').replaceChildren(factsLine(t('features'),spec.features.toLocaleString(ViewerLanguage.locale)),factsLine(t('activity'),spec.k),factsLine(t('grid'),meta.native_shape.join(' × ')),factsLine(t('patient'),$('case').selectedOptions[0].textContent));
  $('response-status').textContent=meta.pooled ? `${t('pooled')}: ${meta.maximum.toFixed(4)}` : `${t('maximum')}: ${meta.maximum.toFixed(4)} · ${t('mean')}: ${meta.mean.toFixed(4)} · ${t('fraction')}: ${(meta.active_fraction*100).toFixed(1)}%${meta.maximum===0 ? ` · ${t('inactiveMap')}` : ''}`;
  $('resolution').textContent=meta.pooled ? t('pooled') : `${t('grid')}: ${meta.native_shape.join(' × ')} · ${t('interpolation')}`;
  $('pooled-note').hidden=!meta.pooled;$('pooled-note').textContent=meta.pooled?t('pooledNote'):'';
  const rows=[[t('fvu'),spec.development.fvu.toFixed(5)],[t('cosine'),spec.development.cosine.toFixed(5)],[t('mae'),spec.downstream.probability_mae.toFixed(5)],[t('auc'),spec.downstream.recovered_auroc.toFixed(5)],[t('inactive'),`${(spec.development.inactive_fraction*100).toFixed(2)}%`],[t('hash'),spec.dictionary_sha256]];
  $('metrics').replaceChildren(...rows.flatMap(([label,value])=>[node('dt',label),node('dd',value)]));
  $('run-id').textContent=spec.run_id;
}
async function fetchValue(path, signal, binary=false) {
  const response=await fetch(path,{signal});
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return binary ? new Float32Array(await response.arrayBuffer()) : response.json();
}
function clearCanvases() {
  for (const axis of [0,1,2]) {const canvas=$(`view-${axis}`);canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);}
}
async function load(resetScale=false) {
  state.request?.abort();
  const controller=new AbortController();state.request=controller;
  state.error=null;state.loadingKey='loading';$('error').hidden=true;$('loading').textContent=t('loading');$('viewports').setAttribute('aria-busy','true');
  $('pooled-note').hidden=true;$('facts').replaceChildren();$('response-status').textContent='';
  $('metrics').replaceChildren();$('run-id').textContent='';$('resolution').textContent='';
  state.meta=null;state.volume=null;state.map=null;clearCanvases();
  const spec=state.catalog.dictionaries.find(row=>row.run_id===$('dictionary').value);
  state.spec=spec;
  const feature=Math.max(0,Math.min(spec.features-1,Math.trunc(Number($('feature').value)||0))),caseId=Number($('case').value);
  $('feature').value=feature;$('feature').max=spec.features-1;$('previous').disabled=feature===0;$('next').disabled=feature===spec.features-1;
  const base=`/api/dictionaries/${encodeURIComponent(spec.run_id)}/${caseId}/${feature}`;
  try {
    const [meta,volume,map]=await Promise.all([fetchValue(base,controller.signal),fetchValue(`/api/dictionary-volume/${spec.model}/${caseId}`,controller.signal,true),fetchValue(`${base}/map`,controller.signal,true)]);
    if (controller.signal.aborted) return;
    if (volume.length!==meta.shape.reduce((a,b)=>a*b,1) || map.length!==(meta.pooled?1:volume.length)) throw new Error('Image array size mismatch');
    state.meta=meta;state.volume=volume;state.map=map;state.position=meta.shape.map(size=>Math.floor(size/2));
    const scaleKey=`${spec.run_id}/${feature}`;
    if(resetScale || state.scaleKey!==scaleKey) {$('scale').value=Math.max(meta.reference_maximum,0.001).toPrecision(5);state.scaleKey=scaleKey;}
    $('overlay').disabled=meta.pooled;$('peak').disabled=meta.pooled;$('opacity').disabled=meta.pooled;$('scale').disabled=meta.pooled;
    for(const axis of [0,1,2]) {$(`slice-${axis}`).min=0;$(`slice-${axis}`).max=meta.shape[axis]-1;}
    url.searchParams.set('run',spec.run_id);url.searchParams.set('case',caseId);url.searchParams.set('feature',feature);history.replaceState(null,'',url);
    $('viewports').setAttribute('aria-busy','false');showFacts();draw();state.loadingKey=null;$('loading').textContent='';
  } catch(error) {
    if(error.name==='AbortError') return;
    state.loadingKey=null;state.error=error.message;$('loading').textContent='';$('viewports').setAttribute('aria-busy','false');$('error').textContent=`${t('error')} ${error.message}`;$('error').hidden=false;
  }
}
function axesFor(axis) {return axis===2?[0,1]:axis===1?[0,2]:[1,2];}
function color(value) {
  const stops=[[19,63,197],[0,200,222],[252,246,71],[242,61,25]],scaled=Math.max(0,Math.min(.999999,value))*3,index=Math.floor(scaled),fraction=scaled-index;
  return stops[index].map((channel,i)=>channel*(1-fraction)+stops[index+1][i]*fraction);
}
function draw() {
  if(!state.meta) return;
  const {shape,spacing,pooled}=state.meta;
  const windows={lung:[-1000,400],tissue:[-160,240],bone:[-500,1500]},range=windows[$('window').value];
  const scale=Math.max(.001,Number($('scale').value)||.001),opacity=Number($('opacity').value)/100,overlay=$('overlay').checked&&!pooled;
  $('legend-max').textContent=scale.toPrecision(4);
  for (const axis of [0,1,2]) {
    const canvas=$(`view-${axis}`),ctx=canvas.getContext('2d'),rect=canvas.getBoundingClientRect(),ratio=devicePixelRatio||1;
    const width=Math.max(1,Math.round(rect.width*ratio)),height=Math.max(1,Math.round(rect.height*ratio));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    const [horizontal,vertical]=axesFor(axis),columns=shape[horizontal],rows=shape[vertical];
    const image=new ImageData(columns,rows);
    for(let y=0;y<rows;y++) for(let x=0;x<columns;x++) {
      const point=[...state.position];point[horizontal]=x;point[vertical]=rows-1-y;
      const index=(point[0]*shape[1]+point[1])*shape[2]+point[2],pixel=(y*columns+x)*4;
      const gray=Math.max(0,Math.min(255,(state.volume[index]-range[0])/(range[1]-range[0])*255));
      let channels=[gray,gray,gray];
      if(overlay&&state.map[index]>0) {const heat=color(state.map[index]/scale),alpha=opacity*Math.min(1,state.map[index]/scale*2);channels=channels.map((channel,i)=>channel*(1-alpha)+heat[i]*alpha);}
      image.data.set([...channels,255],pixel);
    }
    const off=document.createElement('canvas');off.width=columns;off.height=rows;off.getContext('2d').putImageData(image,0,0);
    const factor=Math.min(width/(columns*spacing[horizontal]),height/(rows*spacing[vertical])),w=columns*spacing[horizontal]*factor,h=rows*spacing[vertical]*factor,left=(width-w)/2,top=(height-h)/2;
    ctx.fillStyle='#071014';ctx.fillRect(0,0,width,height);ctx.imageSmoothingEnabled=false;ctx.drawImage(off,left,top,w,h);
    state.rectangles[axis]={left:left/ratio,top:top/ratio,width:w/ratio,height:h/ratio,columns,rows};
    ctx.strokeStyle='rgba(155,237,214,.85)';ctx.lineWidth=ratio;
    const px=left+(state.position[horizontal]+.5)/columns*w,py=top+(rows-state.position[vertical]-.5)/rows*h;
    ctx.beginPath();ctx.moveTo(px,top);ctx.lineTo(px,top+h);ctx.moveTo(left,py);ctx.lineTo(left+w,py);ctx.stroke();
    $(`slice-${axis}`).value=state.position[axis];$(`position-${axis}`).textContent=`${state.position[axis]+1} / ${shape[axis]}`;
  }
}
function move(axis,delta) {if(!state.meta)return;state.position[axis]=Math.max(0,Math.min(state.meta.shape[axis]-1,state.position[axis]+delta));draw();}
function dictionaryChanged() {
  const spec=state.catalog.dictionaries.find(row=>row.run_id===$('dictionary').value),previous=$('case').value;
  $('case').replaceChildren(...state.catalog.cases[spec.model].map(row=>{const option=node('option',row.label);option.value=row.index;return option;}));
  if([...$('case').options].some(option=>option.value===previous)) $('case').value=previous;
  load(true);
}
async function initialize() {
  state.loadingKey='loadCatalog';state.error=null;$('loading').textContent=t('loadCatalog');$('error').hidden=true;
  try {
    state.catalog=await fetchValue('/api/dictionaries');
    const available=state.catalog.dictionaries.length>0;
    $('empty').hidden=available;$('workspace').hidden=!available;
    if(!available){state.loadingKey=null;$('loading').textContent='';return;}
    $('dictionary').replaceChildren(...state.catalog.dictionaries.map(row=>{const option=node('option',`${row.model==='vista'?'VISTA3D':'FMCIB'} · ${row.site} · k${row.k} · seed ${row.seed}`);option.value=row.run_id;return option;}));
    if(state.catalog.dictionaries.some(row=>row.run_id===url.searchParams.get('run'))) $('dictionary').value=url.searchParams.get('run');
    const spec=state.catalog.dictionaries.find(row=>row.run_id===$('dictionary').value);
    $('case').replaceChildren(...state.catalog.cases[spec.model].map(row=>{const option=node('option',row.label);option.value=row.index;return option;}));
    const caseId=url.searchParams.get('case');if([...$('case').options].some(option=>option.value===caseId)) $('case').value=caseId;
    $('feature').value=url.searchParams.get('feature')||0;
    await load(true);
  } catch(error) {state.loadingKey=null;state.error=error.message;$('loading').textContent='';$('error').textContent=`${t('error')} ${error.message}`;$('error').hidden=false;}
}
for(const axis of [0,1,2]) {
  $(`slice-${axis}`).addEventListener('input',event=>{state.position[axis]=Number(event.target.value);draw();});
  const canvas=$(`view-${axis}`);
  canvas.addEventListener('wheel',event=>{event.preventDefault();move(axis,Math.sign(event.deltaY));},{passive:false});
  canvas.addEventListener('keydown',event=>{if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();move(axis,['ArrowUp','ArrowRight'].includes(event.key)?1:-1);}});
  canvas.addEventListener('click',event=>{
    if(!state.meta)return;
    const rect=canvas.getBoundingClientRect(),view=state.rectangles[axis],[horizontal,vertical]=axesFor(axis),x=event.clientX-rect.left,y=event.clientY-rect.top;
    state.position[horizontal]=Math.max(0,Math.min(view.columns-1,Math.floor((x-view.left)/view.width*view.columns)));
    state.position[vertical]=Math.max(0,Math.min(view.rows-1,view.rows-1-Math.floor((y-view.top)/view.height*view.rows)));
    draw();
  });
}
$('dictionary').addEventListener('change',dictionaryChanged);
$('case').addEventListener('change',()=>load());$('feature').addEventListener('change',()=>load(true));
$('strongest').addEventListener('click',()=>{if(state.meta){$('case').value=state.meta.strongest_case;load();}});
for(const [id,delta] of [['previous',-1],['next',1]]) $(id).addEventListener('click',()=>{$('feature').value=Number($('feature').value)+delta;load(true);});
for(const id of ['window','overlay','opacity','scale']) $(id).addEventListener('input',draw);
$('peak').addEventListener('click',()=>{if(state.meta){state.position=[...state.meta.peak];draw();}});
$('center').addEventListener('click',()=>{if(state.meta){state.position=state.meta.shape.map(size=>Math.floor(size/2));draw();}});
$('refresh').addEventListener('click',initialize);
window.addEventListener('viewer-language-change',language);window.addEventListener('resize',draw);
language();initialize();
