'use strict';
const el = id => document.getElementById(id);
const names = {fmcib:'FMCIB',vista:'VISTA3D'};
const labels = {id:'Feature 编号',maximum:'最大响应',lung_enrichment:'肺部富集程度',liver_enrichment:'肝脏富集程度',outside_fraction:'图像边缘空气响应',high_hu_fraction:'高 HU 区域的响应',reappearance:'跨 seed 重现次数',nodule_enrichment:'结节区域富集',malignancy_partial:'控制大小后的恶性关联（discovery）',image_r2:'图像变量解释程度',frequency:'激活频率'};
const groupLabels = {interpretation:'解释用字典 · 训练后 encoder',control:'随机权重 encoder 对照'};
const stabilityLabels = {stable:'稳定',partial:'部分重现',unstable:'不稳定',inactive:'开发集未激活'};
const regionLabels = {inside:'结节内部',margin_0_4mm:'表面外 0–4 mm',near_4_10mm:'表面外 4–10 mm',far_over_10mm:'表面外 10 mm 以上'};
const memories = JSON.parse(localStorage.getItem('sclvmi-heatmap-selections')||'{}');
const state = {catalog:[],run:null,scope:'nodule',feature:0,case:0,mode:'spatial',point:null,pointKey:null,meta:null,features:[],cases:[],page:0,request:0,space:false,pan:{},drag:null,rects:{},scaleKey:null,mask:null,detail:null};
const pageSize=80,volumeCache=new Map();
let renderer,controller,drawFrame=0;
const format=value=>Number(value).toLocaleString(ViewerLanguage.locale,{maximumFractionDigits:3});
const text=value=>ViewerLanguage.value==='en'?ViewerLanguage.text(value):value;
const node=(tag,value,className)=>{const item=document.createElement(tag);if(value!=null)item.textContent=value;if(className)item.className=className;return item;};
const spec=()=>state.catalog.find(row=>row.run_id===state.run);
const source=()=>state.scope==='nodule'?spec().image_source:spec().model;
const interpretive=()=>state.scope==='nodule';
const siteName=row=>row.site==='layer1_context'?`layer1 · ${text('完整上下文')}`:row.site;
const configurationName=row=>`${names[row.model]} · ${siteName(row)} · e${row.expansion} · k${row.k}`;
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
  const url=`/whole?${query()}`;
  if(push&&`${location.pathname}${location.search}`!==url)history.pushState(null,'',url);else history.replaceState(null,'',url);
  localStorage.setItem('sclvmi-heatmap-location',url);remember();
}
function loading(message,error=false){
  const shell=document.querySelector('.viewer-shell');shell.setAttribute('aria-busy',String(!error));shell.toggleAttribute('data-error',error);
  el('viewer-loading').hidden=false;el('viewer-loading-text').textContent=text(message);document.querySelector('.viewer-grid').style.visibility='hidden';state.meta=null;state.rects={};
  for(const axis of [0,1,2]){const canvas=el(`canvas-${axis}`);canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);el(`slice-${axis}`).disabled=true;el(`position-${axis}`).textContent='';}
  for(const id of ['peak','center','strongest','reset-scale'])el(id).disabled=true;
  for(const id of ['map-stats','point-value','map-label','geometry-note','case-label','stability-summary','stability-table','region-bars','region-note','malignancy-values','malignancy-note','control-values','control-note','evidence-scope','autointerp-routes','autointerp-note','autointerp-examples'])el(id).replaceChildren();
  state.detail=null;
}
function report(error){if(error.name==='AbortError')return;el('loading').textContent='';el('error-message').textContent=error.message;el('error').hidden=false;loading('影像读取失败。可以重新读取，或选择另一份图像。',true);}
function run(action){Promise.resolve().then(action).catch(report);}
function modes(){
  if(interpretive())return [['spatial','空间激活 · 插值显示'],['native','空间激活 · 原始采样']];
  return [['spatial','空间激活 · 原始采样'],['single','每个区域 · 中央位置'],['mean','每个区域 · 平均值'],['maximum','每个区域 · 峰值']];
}
function controls(){
  const current=spec(),families=[...new Set(state.catalog.filter(row=>row.scopes.includes(state.scope)).map(row=>row.family))];
  for(const option of el('category').options)option.textContent=text(labels[option.value]);
  el('model').replaceChildren();
  for(const name of ['interpretation','control']){
    const group=node('optgroup');group.label=text(groupLabels[name]);
    for(const family of families){const row=state.catalog.find(item=>item.family===family);if(row.group!==name)continue;const option=node('option',configurationName(row));option.value=family;group.append(option);}
    if(group.children.length)el('model').append(group);
  }
  el('model').value=current.family;
  el('seed').replaceChildren(...state.catalog.filter(row=>row.family===current.family&&row.scopes.includes(state.scope)).map(row=>{const option=node('option',row.seed);option.value=row.run_id;return option;}));
  el('seed').value=state.run;el('seed').disabled=el('seed').options.length===1;el('scope').value=state.scope;
  const available=modes();if(!available.some(([value])=>value===state.mode))state.mode=available[0][0];
  el('mode').replaceChildren(...available.map(([value,label])=>{const option=node('option',text(label));option.value=value;return option;}));el('mode').value=state.mode;
  const encoderName=current.group==='control'?` · ${text('随机权重')}`:'';
  el('dictionary-stamp').textContent=`${names[current.model]} · ${siteName(current)}${encoderName} · seed ${current.seed}`;
  el('dictionary-note').textContent=`${format(current.features)} features · k${current.k} · seed ${current.seed}. ${text('Feature 编号属于当前字典。')}${current.nodule_interpretation?'':' '+text('此位置不用于结节解释。')}`;
  el('feature-title').textContent=`${names[current.model]} · ${siteName(current)}${encoderName} · Feature ${state.feature}`;el('feature-description').textContent=state.features.find(row=>row.id===state.feature)?.description||'';
  el('cases-title').textContent=text(interpretive()?'结节解释病例':'完整 CT 病例');
  el('cases-note').textContent=text(interpretive()?'D：128 名训练集患者（discovery）；C：128 名开发集患者（confirmation）':'每组第一例用于观察，第二例用于确认');
  el('dataset-note').textContent=interpretive()?'LUNA25 · training + development':'PETWB · training';el('source-note').textContent='20261014 · '+text(groupLabels[current.group]);
  el('scope-help').textContent=text(interpretive()?(current.site==='layer1_context'?'FMCIB layer1 使用以结节为中心的 84 mm 输入，只保留 35 体素感受野完全位于原始视野内的位置；感受野触及视野外的位置不显示响应。绿色轮廓为结节 mask。':'该字典使用模型部署时的输入窗口；窗口上下文会影响响应。绿色轮廓为结节 mask。'):'在完整 CT 上使用与训练相同大小的输入窗口：FMCIB layer1 完整上下文为 84 mm、每隔 48 mm，只保留感受野完全位于 CT 内的位置；FMCIB layer2 为 50 mm、每隔 32 mm；VISTA3D 为 72 mm、每隔 48 mm。相邻窗口的中央区域拼接成完整图，窗口上下文可能影响相邻区域的响应。');
  el('mask-control').hidden=!interpretive();
  const pool=current.nodule_pool_fraction,mixture=pool?`${text('结节表面 10 mm 以内')} ${Math.round(pool*100)}% · ${text('均匀抽样')} ${Math.round((1-pool)*100)}%`:text('均匀抽样');
  const quality=current.quality,rows=[['输入位置',text(current.site==='layer1_context'?'完整感受野位置':'部署输入窗口')],['训练批次组成',mixture],['训练步数',`${format(current.training_steps)} × 2,048`],['开发集 FVU',quality.fvu.toFixed(4)],['同系数 PCA 的 FVU',quality.pca_fvu.toFixed(4)],['稳定 Feature',`${format(quality.stable_features)} / ${format(quality.live_features)}`],['稳定 Feature 解释的方差',quality.stable_explained_variance.toFixed(4)],['decoder 配对相关（PW-MCC）',quality.pw_mcc.toFixed(4)]];
  if(current.downstream)rows.push(['分类概率 MAE',current.downstream.probability_mae.toFixed(5)],['重建后 AUROC',`${current.downstream.recovered_auroc.toFixed(4)} / ${current.downstream.original_auroc.toFixed(4)}`]);
  if(current.interior_downstream)rows.push(['只替换完整感受野位置的 MAE',current.interior_downstream.probability_mae.toFixed(5)]);
  el('quality-values').replaceChildren(...rows.flatMap(([label,value])=>[node('dt',text(label)),node('dd',value)]));
  el('quality-scope').textContent=text(current.group==='control'?'随机权重 encoder 使用相同的结构、预处理、采样和 SAE 配置；它没有分类器，用于判断哪些 Feature 性质超过随机网络对 CT 灰度的响应。':current.nodule_interpretation?'该位置按结节覆盖标准选择：训练批次的一部分来自结节内部、表面外 0–4 mm 和 4–10 mm 的位置（三段各占三分之一），其余为均匀抽样；8 个 seed 都通过固定分类器检查。数值来自训练集与开发集；独立测试集未使用。':'该位置保留均匀抽样训练的字典，不用于结节解释。数值来自训练集与开发集；独立测试集未使用。');
  el('dictionary-identity').textContent=`${current.run_id} · SHA-256 ${current.dictionary_sha256}`;
}
function featureMetric(row,key){
  if(Number.isFinite(row[key]))return {rank:row[key],label:format(row[key])};
  const region=key==='lung_enrichment'?'lung':key==='liver_enrichment'?'liver':null;
  if(region&&row.discovery_region_means?.[region]>0)return {rank:Infinity,label:text('参考区域均值为零')};
  if(['nodule_enrichment','malignancy_partial','image_r2','frequency'].includes(key))return {rank:-Infinity,label:text('无数据')};
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
    if(row.stability){const total=state.comparisons||7;button.append(node('small',`${text(stabilityLabels[row.stability])} · ${row.reappearance}/${total}`,`stability-tag ${row.stability}`));}
    const key=el('category').value;if(key!=='id'&&key!=='reappearance')button.append(node('small',`${text(labels[key])} ${featureMetric(row,key).label}`));return button;
  }));
  list.scrollTop=scroll;if(focused)list.querySelector(`[data-feature="${focused}"]`)?.focus({preventScroll:true});
  el('feature-count').textContent=format(state.features.length);el('empty-features').hidden=rows.length!==0;el('page-label').textContent=`${state.page+1} / ${pages}`;el('previous-page').disabled=state.page===0;el('next-page').disabled=state.page===pages-1;
  if(locate){const selected=list.querySelector(`[data-feature="${state.feature}"]`);if(selected){const bounds=list.getBoundingClientRect(),item=selected.getBoundingClientRect();if(getComputedStyle(list).display==='flex'){if(item.left<bounds.left)list.scrollLeft-=Math.ceil(bounds.left-item.left);else if(item.right>bounds.right)list.scrollLeft+=Math.ceil(item.right-bounds.right);}else if(item.top<bounds.top)list.scrollTop-=Math.ceil(bounds.top-item.top);else if(item.bottom>bounds.bottom)list.scrollTop+=Math.ceil(item.bottom-bounds.bottom);}}
}
function updateCases(){
  const current=spec();
  const key=`${state.scope}/${source()}/${ViewerLanguage.value}/${state.cases.map(row=>Number(row.ready)).join('')}`;
  if(el('case-strip').dataset.catalog!==key){
    const caseText=row=>row.malignant==null?row.split:`${row.split} · ${text(row.malignant?'恶性':'良性')} · ${format(row.diameter_mm)} mm`;
    el('case-select').replaceChildren(...state.cases.map(row=>{const option=node('option',`${row.label} · ${caseText(row)}${row.ready?'':` · ${text('准备中')}`}`);option.value=row.id;return option;}));
    el('case-strip').replaceChildren(...state.cases.map(row=>{const button=node('button',null,'case-tile');button.dataset.case=row.id;const image=node('img');image.className='whole-thumb';image.src=`/api/heatmaps/thumbnail/${state.scope}/${source()}/${row.id}`;image.alt=`${row.label} ${text('冠状面 CT')}`;image.loading='lazy';button.append(image,node('strong',row.label),node('span',caseText(row)));return button;}));el('case-strip').dataset.catalog=key;
  }
  el('case-select').value=state.case;for(const button of el('case-strip').children){const active=Number(button.dataset.case)===state.case;button.classList.toggle('active',active);button.setAttribute('aria-current',String(active));}
  const currentCase=state.cases[state.case];
  el('case-label').textContent=currentCase?`${currentCase.label} · ${currentCase.split}${currentCase.malignant==null?'':` · ${text(currentCase.malignant?'恶性':'良性')} · ${text('最大直径')} ${format(currentCase.diameter_mm)} mm`}`:'';
}
function showMetadata(){
  const meta=state.meta;if(!meta)return;const spacing=Array.isArray(meta.map_spacing_mm)?meta.map_spacing_mm.map(format).join(' × '):format(meta.map_spacing_mm);
  el('map-label').textContent=`${el('mode').selectedOptions[0].textContent} · ${spacing} mm`;el('geometry-note').textContent=text(meta.scale_source);
  const rows=[['最大值',format(meta.maximum)],['平均值',format(meta.mean)],['有响应的位置',`${format(meta.nonzero_fraction*100)}%`],['真实采样网格',(meta.native_shape||meta.map_shape).join(' × ')]];el('map-stats').replaceChildren(...rows.flatMap(([label,value])=>[node('dt',text(label)),node('dd',value)]));
}
function evidenceRows(target,rows){target.replaceChildren(...rows.flatMap(([label,value])=>[node('dt',text(label)),node('dd',value)]));}
function showEvidence(){
  const detail=state.detail;el('interpretation-evidence').hidden=!detail;if(!detail)return;
  const percent=value=>value==null?text('无数据'):`${format(value)}%`,number=value=>value==null?text('无数据'):format(value);
  el('stability-summary').textContent=`${text(stabilityLabels[detail.stability])} · ${text('在其他 seed 中重现')} ${detail.reappearance} / ${detail.comparisons} · decoder cosine ≥ ${detail.cosine_threshold}`;
  const head=node('tr');for(const label of ['Seed','Feature','decoder cosine','激活相关'])head.append(node('th',text(label)));
  el('stability-table').replaceChildren(node('thead'),node('tbody'));el('stability-table').tHead.append(head);
  for(const match of detail.matches){
    const row=node('tr',null,match.decoder_cosine>=detail.cosine_threshold?'matched':'');row.append(node('td',match.seed));
    const cell=node('td');if(match.run_id){const button=node('button',String(match.feature),'text-button');button.onclick=()=>{remember();state.run=match.run_id;state.feature=match.feature;el('search').value='';state.page=0;run(()=>loadSelection({catalog:true,locate:true,push:true}));};cell.append(button);}else cell.textContent=match.feature;
    row.append(cell,node('td',number(match.decoder_cosine)),node('td',number(match.activation_correlation)));el('stability-table').tBodies[0].append(row);
  }
  const values=Object.values(detail.region_mean).filter(value=>value!=null),highest=Math.max(...values,1e-9);
  el('region-bars').replaceChildren(...Object.entries(detail.region_mean).map(([key,value])=>{const item=node('div',null,'region-bar'),bar=node('span');bar.style.width=`${Math.max(0,(value||0)/highest*100)}%`;item.append(node('strong',text(regionLabels[key])),node('div',null,'bar-track'),node('small',number(value)));item.querySelector('.bar-track').append(bar);return item;}));
  el('region-note').textContent=`${text('全部有效位置平均')} ${number(detail.overall_mean)} · ${text('结节区域富集')} ${number(detail.nodule_enrichment)} · ${text('激活频率')} ${percent(detail.frequency==null?null:detail.frequency*100)}`;
  evidenceRows(el('malignancy-values'),[['单独使用该 Feature 的开发集 AUROC',number(detail.development_auroc)],['控制大小后的 partial correlation · discovery',number(detail.partial_train)],['控制大小后的 partial correlation · confirmation',number(detail.partial_development)],['仅用结节体积的开发集 AUROC',number(detail.size_only_development_auroc)]]);
  el('malignancy-note').textContent=text('病例数值为结节内部及表面外 4 mm 以内有效位置的最大响应；partial correlation 为控制对数体积后的 Spearman partial correlation。');
  const r2=detail.image_r2==null&&spec().seed!==2025?text('只在参照 seed 2025 计算'):number(detail.image_r2);
  if(detail.encoder==='trained'){
    evidenceRows(el('control-values'),[['结节区域富集超过的随机 Feature 比例',percent(detail.enrichment_percentile)],['恶性关联绝对值超过的随机 Feature 比例',percent(detail.partial_percentile)],['图像变量解释程度 R²',r2]]);
    el('control-note').textContent=text('比较对象是相同结构、随机权重 encoder 上训练的参照 seed 字典的全部 Feature。图像变量包括局部 HU 均值、标准差、梯度、空气与高密度比例、周围 HU 均值和到结节表面的距离。');
  }else{
    evidenceRows(el('control-values'),[['图像变量解释程度 R²',r2]]);
    el('control-note').textContent=text('当前为随机权重 encoder 的对照字典。');
  }
  el('evidence-scope').textContent=`${text('统计使用每名患者一个结节记录')}：discovery ${detail.discovery_patients} · confirmation ${detail.confirmation_patients}。${text('这些是描述性统计，医学含义需要独立证据；独立测试集未使用。')}`;
  showAutointerp(detail.autointerp,number);
}
function showAutointerp(auto,number){
  el('autointerp-card').hidden=!auto;if(!auto)return;
  const routeNames={describe_summarize:'逐例描述后归纳',direct:'直接从图像归纳'};
  el('autointerp-routes').replaceChildren(...auto.routes.map(route=>{
    const item=node('div',null,'autointerp-route');item.append(node('strong',text(routeNames[route.route])));
    if(route.no_clear_pattern){item.append(node('p',text('模型认为高激活与低激活病例之间没有清楚的共同差异。')));return item;}
    item.append(node('p',`${route.label}: ${route.description}`),node('small',`${text('验证 AUROC')} ${number(route.auroc)} [${number(route.auroc_low)}, ${number(route.auroc_high)}] · ${text('评分均值（高激活 / 低激活）')} ${number(route.mean_score_high)} / ${number(route.mean_score_low)}`));
    return item;
  }));
  el('autointerp-note').textContent=text('模型 {model}。验证使用开发集中激活最高的 {high} 名患者与大小相近的 {low} 名低激活患者，评分只看原始 CT，不看激活；结节测量基线 AUROC {baseline}。描述由通用视觉模型生成，医学名称需要人工核验。下方为生成解释时使用的训练集病例图像。')
    .replace('{model}',auto.model).replace('{high}',auto.validation.high).replace('{low}',auto.validation.low).replace('{baseline}',number(auto.measurement_auroc));
  el('autointerp-examples').replaceChildren(...auto.examples.map(example=>{
    const link=node('a',null,'autointerp-thumb');link.href=`/api/heatmaps/autointerp/${encodeURIComponent(auto.key)}/${encodeURIComponent(example.id)}`;link.target='_blank';link.rel='noopener';
    const image=node('img');image.src=link.href;image.alt=example.id;image.loading='lazy';link.append(image,node('small',`${example.id} · ${text(example.group==='high'?'高激活':'低激活')}`));return link;
  }));
}
async function loadSelection({catalog=false,locate=false,push=false,resetPoint=false}={}){
  catalog=catalog||!state.features.length;controller?.abort();controller=new AbortController();const signal=controller.signal,id=++state.request,started=performance.now();
  if(state.windowScope!==state.scope){el('window').value=state.scope==='nodule'?'lung':'tissue';state.windowScope=state.scope;}
  if(catalog){state.features=[];state.cases=[];el('feature-list').replaceChildren();el('feature-list').inert=true;el('case-select').replaceChildren();el('case-strip').replaceChildren();delete el('case-strip').dataset.catalog;el('feature-count').textContent='';el('page-label').textContent='…';el('previous-page').disabled=true;el('next-page').disabled=true;}
  controls();loading(`正在加载 ${names[spec().model]} · Feature ${state.feature}…`);el('error').hidden=true;el('loading').textContent=text('读取影像与响应…');
  try{
    if(catalog){
      const data=await get(`/api/heatmaps/features?${new URLSearchParams({run:state.run,scope:state.scope})}`,signal);if(id!==state.request)return;state.features=data.features;state.cases=data.cases;state.comparisons=state.catalog.filter(row=>row.family===spec().family).length-1;
      const order=el('category').value;el('category').replaceChildren(...data.rankings.map(key=>{const option=node('option',text(labels[key]));option.value=key;return option;}));if(data.rankings.includes(order))el('category').value=order;el('feature-list').inert=false;
    }
    if(!Number.isInteger(state.feature)||state.feature<0||state.feature>=spec().features)throw new Error(text('Feature 编号超出当前字典范围。'));
    if(!Number.isInteger(state.case)||state.case<0||state.case>=state.cases.length)throw new Error(text('病例编号超出当前图像范围。'));
    controls();syncURL(push);featureList(locate);updateCases();
    if(!state.cases[state.case].ready)throw new Error(text('当前病例的完整 CT 浏览文件尚未生成。'));
    const selection=query().toString(),current=spec();
    const [meta,ct,activity,mask,detail]=await Promise.all([get(`/api/heatmaps/map?${selection}`,signal),get(`/api/heatmaps/ct/${state.scope}/${source()}/${state.case}`,signal,true),get(`/api/heatmaps/activity?${selection}`,signal,true),interpretive()?get(`/api/heatmaps/mask/nodule/${source()}/${state.case}`,signal,true):null,get(`/api/heatmaps/detail?${new URLSearchParams({run:state.run,feature:state.feature})}`,signal)]);if(id!==state.request)return;
    if(!renderer)renderer=new WholeVolumeRenderer();renderer.upload(meta,ct,activity,state.mode==='native'?'spatial':state.mode);state.meta=meta;state.pan={};state.mask=mask;state.detail=detail;
    const pointKey=`${state.scope}/${current.model}/${state.case}`;if(resetPoint||state.pointKey!==pointKey||!state.point)state.point=meta.shape.map(size=>Math.floor(size/2));state.point=state.point.map((value,axis)=>Math.max(0,Math.min(meta.shape[axis]-1,value)));state.pointKey=pointKey;
    const scaleKey=`${state.scope}/${state.run}/${state.feature}/${state.mode}`;if(scaleKey!==state.scaleKey){el('scale').value=Number(meta.scale.toPrecision(5));state.scaleKey=scaleKey;}
    for(let axis=0;axis<3;axis++){el(`slice-${axis}`).max=meta.shape[axis]-1;el(`slice-${axis}`).value=state.point[axis];el(`slice-${axis}`).disabled=false;}
    for(const name of ['reset-scale','peak','center'])el(name).disabled=false;el('strongest').hidden=state.scope==='whole';el('strongest').disabled=meta.strongest_case==null;
    showMetadata();showEvidence();
    paintAll();document.querySelector('.viewer-grid').style.visibility='visible';el('viewer-loading').hidden=true;document.querySelector('.viewer-shell').setAttribute('aria-busy','false');el('loading').textContent='';document.querySelector('.viewer-shell').dataset.selection=selection;document.querySelector('.viewer-shell').dataset.loadMs=String(performance.now()-started);
  }catch(error){if(id===state.request)report(error);}
}
function draw(axis){
  const meta=state.meta;if(!meta||!renderer)return;const canvas=el(`canvas-${axis}`),box=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;if(!box.width||!box.height)return;
  const widthPx=Math.round(box.width*dpr),heightPx=Math.round(box.height*dpr);if(canvas.width!==widthPx||canvas.height!==heightPx){canvas.width=widthPx;canvas.height=heightPx;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#0b151a';ctx.fillRect(0,0,box.width,box.height);
  const [u,v]=[0,1,2].filter(value=>value!==axis),nx=meta.shape[u],ny=meta.shape[v],physical=[nx*meta.spacing[u],ny*meta.spacing[v]],zoom=Number(el('zoom').value),unit=Math.min((box.width-32)/physical[0],(box.height-32)/physical[1])*zoom;
  const width=physical[0]*unit,height=physical[1]*unit,pan=state.pan[axis]||[0,0],left=(box.width-width)/2+pan[0],top=(box.height-height)/2+pan[1];state.rects[axis]={left,top,width,height,nx,ny,u,v};
  const windows={tissue:[-160,240],lung:[-1000,400],bone:[-500,1500]},scale=Math.max(.001,Number(el('scale').value)),opacity=Number(el('opacity').value)/100,threshold=Number(el('threshold').value)/100,overlay=el('overlay').checked&&!state.space;
  const off=renderer.render(axis,state.point[axis],{window:windows[el('window').value],scale,opacity:overlay?opacity:0,threshold});ctx.imageSmoothingEnabled=false;ctx.drawImage(off,left,top,width,height);
  if(state.mask&&interpretive()&&el('mask-outline').checked&&!state.space){
    const shape=meta.shape,cw=width/nx,ch=height/ny,position=state.point[axis];
    const value=(i,j)=>{if(i<0||j<0||i>=nx||j>=ny)return 0;const p=[0,0,0];p[axis]=position;p[u]=i;p[v]=j;return state.mask[(p[0]*shape[1]+p[1])*shape[2]+p[2]];};
    ctx.beginPath();
    for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){
      if(value(i,j)<.5)continue;const x0=left+i*cw,y0=top+(ny-1-j)*ch;
      if(value(i-1,j)<.5){ctx.moveTo(x0,y0);ctx.lineTo(x0,y0+ch);}if(value(i+1,j)<.5){ctx.moveTo(x0+cw,y0);ctx.lineTo(x0+cw,y0+ch);}
      if(value(i,j-1)<.5){ctx.moveTo(x0,y0+ch);ctx.lineTo(x0+cw,y0+ch);}if(value(i,j+1)<.5){ctx.moveTo(x0,y0);ctx.lineTo(x0+cw,y0);}
    }
    ctx.strokeStyle='rgba(116,240,140,.95)';ctx.lineWidth=1.5;ctx.stroke();
  }
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
el('scope').onchange=()=>{
  const scope=el('scope').value,current=spec();
  const next=current.scopes.includes(scope)?current:state.catalog.find(row=>row.family===current.family&&row.scopes.includes(scope));
  selectDictionary(next.run_id,scope);
};
el('mode').onchange=()=>{state.mode=el('mode').value;run(()=>loadSelection({push:true}));};
function selectCase(value){if(value===state.case)return;state.case=value;run(()=>loadSelection({push:true,resetPoint:true}));}
el('case-select').onchange=()=>selectCase(Number(el('case-select').value));el('case-strip').onclick=event=>{const button=event.target.closest('[data-case]');if(button)selectCase(Number(button.dataset.case));};el('strongest').onclick=()=>{if(state.meta?.strongest_case!=null)selectCase(state.meta.strongest_case);};
for(const id of ['overlay','window','zoom','mask-outline'])el(id).onchange=drawAll;el('scale').onchange=()=>{if(!Number.isFinite(Number(el('scale').value))||Number(el('scale').value)<=0)el('scale').value=state.meta?.scale||1;drawAll();};
for(const id of ['opacity','threshold'])el(id).oninput=()=>{el(`${id}-value`).textContent=`${el(id).value}%`;drawAll();};el('peak').onclick=()=>{if(state.meta)movePoint([...state.meta.peak]);};el('center').onclick=()=>{if(state.meta)movePoint(state.meta.shape.map(size=>Math.floor(size/2)));};el('reset-scale').onclick=()=>{if(state.meta){el('scale').value=Number(state.meta.scale.toPrecision(5));drawAll();}};el('retry').onclick=()=>run(()=>loadSelection({catalog:!state.features.length}));
el('help-toggle').onclick=()=>{el('help').hidden=!el('help').hidden;el('help-toggle').setAttribute('aria-expanded',String(!el('help').hidden));};
document.querySelectorAll('.expand').forEach(button=>button.onclick=()=>{const grid=document.querySelector('.viewer-grid'),view=button.closest('.viewport'),expanded=!grid.classList.contains('expanded');grid.classList.toggle('expanded',expanded);document.querySelectorAll('.viewport').forEach(item=>item.classList.toggle('selected-view',expanded&&item===view));document.querySelectorAll('.expand').forEach(item=>item.textContent=text(expanded&&item===button?'恢复':'放大'));requestAnimationFrame(drawAll);});
document.addEventListener('keydown',event=>{if(event.code==='Space'&&!['INPUT','SELECT','BUTTON','TEXTAREA','A'].includes(document.activeElement.tagName)){event.preventDefault();state.space=true;drawAll();}});document.addEventListener('keyup',event=>{if(event.code==='Space'){state.space=false;drawAll();}});window.addEventListener('blur',()=>{state.space=false;state.drag=null;drawAll();});new ResizeObserver(drawAll).observe(document.querySelector('.viewer-grid'));
function available(params){const row=state.catalog.find(item=>item.run_id===params.get('run'));return Boolean(row)&&row.scopes.includes(params.get('scope'));}
function readLocation(){const params=new URLSearchParams(location.search);state.run=params.get('run')||state.catalog[0].run_id;if(!spec())throw new Error(text('地址中的 SAE 字典不存在。'));state.scope=params.get('scope')||spec().scopes[0];if(!spec().scopes.includes(state.scope))throw new Error(text('此字典没有对应图像范围的材料。'));state.feature=Number(params.get('feature')||0);state.case=Number(params.get('case')||0);state.mode=params.get('mode')||'spatial';}
window.addEventListener('popstate',()=>run(async()=>{readLocation();el('search').value='';await loadSelection({catalog:true,locate:true});}));
window.addEventListener('viewer-language-change',()=>{if(state.catalog.length){controls();featureList();updateCases();showMetadata();showEvidence();drawAll();}document.title=`SCL-VMI · ${text('热图浏览')}`;});
run(async()=>{const data=await get('/api/heatmaps/catalog');state.catalog=data.dictionaries;const saved=localStorage.getItem('sclvmi-heatmap-location');if(!location.search&&saved&&available(new URLSearchParams(saved.split('?')[1])))history.replaceState(null,'',saved);readLocation();await loadSelection({catalog:true,locate:true,resetPoint:true});});
