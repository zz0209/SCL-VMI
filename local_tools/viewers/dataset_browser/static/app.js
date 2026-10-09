"use strict";
const $ = id => document.getElementById(id);
const state = {catalog: null, dataset: null, case: null, detail: null, offset: 0, total: 0, token: 0, listToken: 0, meshToken: 0, controllers: {}, images: {}, rectangles: {}, cursor: [0,0,0], playing: false, meshReady: false, meshElement: null};
const roleName = role => ({proposed_train: tr("拟定训练部分","Proposed training"), proposed_dev: tr("拟定开发验证","Proposed development"), proposed_test: tr("拟定保留测试","Proposed held-out test"), source_training: tr("来源训练数据","Source training"), source_unlabeled: tr("来源未标注数据","Source unlabeled"), provider_validation: tr("提供方 validation","Provider validation"), provider_unlabeled: tr("提供方 imagesTs","Provider imagesTs")})[role];
const format = (value, decimals=2) => Number(value).toLocaleString(language === "en" ? "en-US" : "zh-CN", {maximumFractionDigits:decimals});
const dimensions = (values, decimals=2) => values.map(v=>format(v,decimals)).join(" × ");
const option = (value, name) => new Option(name, value);
function showError(error) { if (error.name === "AbortError") return; $("error").hidden=false; $("error").textContent=tr("读取或显示未完成：","Could not load or display: ")+error.message+tr("。点击病例重试。",". Select the case to retry."); $("loading").textContent=""; }
async function api(url, signal) { const response=await fetch(url,{signal}); if(!response.ok){const body=await response.text();throw new Error(`${response.status} ${body.slice(0,600)}`);} return response.json(); }
function route(kind) {return `/api/${kind}/${encodeURIComponent(state.dataset.id)}/${encodeURIComponent(state.case)}?channel=${$("channel").value || 0}`;}
function labelName(task, value) { const tasks=state.dataset.tasks; const mapping=task==="label" ? Object.values(tasks)[0] : tasks[task]; return mapping?.[value] ?? `${task}=${value}`; }
function labelsText(labels) {const pairs=Object.entries(labels);return pairs.length ? pairs.map(([task,value])=>labelName(task,value)).join(" · ") : tr("未提供分类标签","No class label");}
function stopPlay(){state.playing=false;$("play").textContent=tr("播放切片","Play slices");}
function updateWindowEditing(){const editable=$("window-preset").value==="custom";for(const id of ["window-low","window-high"])$(id).disabled=!editable;}
function cancelImages(){Object.values(state.controllers).forEach(controller=>controller.abort());state.controllers={};}
function clearImages(){Object.values(state.images).forEach(bitmap=>bitmap.close());state.images={};state.rectangles={};for(const axis of [0,1,2]){const canvas=$(`canvas-${axis}`);canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);}}
function clearMesh(){state.meshToken++;state.controllers.mesh?.abort();if(state.meshElement)Plotly.purge(state.meshElement);state.meshElement=null;state.meshReady=false;$("mesh").replaceChildren();}
function clearView(){stopPlay();cancelImages();clearImages();clearMesh();state.token++;state.detail=null;$("case-body").hidden=true;$("loading").textContent="";$("case-title").textContent=tr("选择病例","Select a case");$("case-subtitle").textContent="";}

function populateDataset(preserve=false){
  const saved={label:$("label-filter").value,role:$("role-filter").value,channel:$("channel").value};
  if(!preserve){clearView();state.case=null;state.offset=0;$("search").value="";}
  state.dataset=state.catalog.datasets.find(d=>d.id===$("dataset").value);
  const d=state.dataset;
  $("dataset-summary").textContent=`${format(d.count,0)} ${tr("份影像记录","image records")} · ${Object.values(d.channels).join(" / ")}${d.patient_count ? " · "+format(d.patient_count,0)+tr(" 位患者"," patients") : ""}`;
  $("channel").replaceChildren(...Object.entries(d.channels).map(([value,name])=>option(value,name)));
  if(preserve)$("channel").value=saved.channel;
  $("label-filter").replaceChildren(option("all",tr("全部分类标签","All labels")));$("distribution").replaceChildren();
  for(const [task,counts] of Object.entries(d.label_counts)){
    for(const [value,count] of Object.entries(counts)){
      $("label-filter").add(option(`${task}:${value}`,`${labelName(task,value)} (${format(count,0)})`));
      const row=document.createElement("div");row.className="distribution-row";
      const name=document.createElement("span");name.textContent=labelName(task,value);const number=document.createElement("span");number.textContent=format(count,0);
      const progress=document.createElement("progress");progress.max=d.count;progress.value=count;progress.setAttribute("aria-label",`${name.textContent}: ${count}`);row.append(name,number,progress);$("distribution").append(row);
    }
  }
  const note=document.createElement("p");note.className="secondary";note.textContent=`${tr("已提供 mask：","Masks available: ")}${format(d.mask_count,0)} / ${format(d.count,0)}`;$("distribution").append(note);
  $("role-filter").replaceChildren(option("all",tr("全部病例","All cases")));
  const roles=d.id==="Dataset005_LUNA25" ? ["proposed_train","proposed_dev","proposed_test"] : d.id==="Dataset007_GliomaIDHType" ? ["provider_validation","provider_unlabeled"] : ["source_training","source_unlabeled"];
  roles.forEach(role=>$("role-filter").add(option(role,roleName(role))));
  $("role-filter").value=d.id==="Dataset005_LUNA25" ? "proposed_train" : "all";
  if(preserve){$("label-filter").value=saved.label;$("role-filter").value=saved.role;}
  updateScope();loadList().catch(showError);
}
function updateScope(){const role=$("role-filter").value;$("scope-note").textContent=state.dataset.id==="Dataset005_LUNA25" ? (role==="proposed_test" ? tr("保留测试候选：浏览将记录，请勿据此选择模型、特征或论文案例。","Held-out candidates: views are logged. Do not use these cases to select models, features, or paper examples.") : tr("LUNA25 分组为拟定患者划分，尚未用于正式实验。","LUNA25 groups follow a proposed patient split, not an established experiment split.")) : state.dataset.id==="Dataset007_GliomaIDHType" ? tr("提供方 validation。","Provider validation.") : "";}
async function loadList(){
  const token=++state.listToken;$("case-list").textContent=tr("读取病例列表…","Loading cases…");
  const params=new URLSearchParams({search:$("search").value.trim(),role:$("role-filter").value,label:$("label-filter").value,offset:state.offset,limit:60});
  const result=await api(`/api/cases/${state.dataset.id}?${params}`);if(token!==state.listToken)return;
  state.total=result.total;$("case-count").textContent=`${format(result.total,0)} ${tr("条","records")}`;
  $("case-list").replaceChildren();
  for(const row of result.items){
    const button=document.createElement("button");button.className="case-row"+(row.id===state.case?" active":"");button.dataset.case=row.id;
    const name=document.createElement("span");name.className="case-id";name.textContent=row.id;
    const label=document.createElement("span");label.className="case-label";label.textContent=labelsText(row.labels);
    button.append(name,label);button.onclick=()=>loadCase(row.id).catch(showError);$("case-list").append(button);
  }
  if(!result.items.length){const empty=document.createElement("p");empty.className="empty";empty.textContent=tr("没有符合条件的病例。请修改搜索或筛选条件。","No matching cases. Adjust the search or filters.");$("case-list").append(empty);}
  $("page-info").textContent=`${result.total ? Math.floor(state.offset/60)+1 : 0} / ${Math.ceil(result.total/60)}`;
  $("previous").disabled=state.offset===0;$("next").disabled=state.offset+60>=result.total;
}

async function loadCase(caseId, saved=null){
  stopPlay();cancelImages();clearImages();clearMesh();const token=++state.token;state.case=caseId;state.detail=null;
  $("error").hidden=true;$("loading").textContent=tr("读取影像与标注…","Loading image and mask…");$("case-body").hidden=true;
  $("case-title").textContent=caseId;$("case-subtitle").textContent=state.dataset.name;
  document.querySelectorAll(".case-row").forEach(row=>row.classList.toggle("active",row.dataset.case===caseId));
  const result=await api(route("case"));if(token!==state.token)return;
  state.detail=result;state.cursor=saved ? saved.cursor : [...result.center];$("case-body").hidden=false;
  $("case-subtitle").textContent=`${state.dataset.name} · ${labelsText(result.labels)} · ${roleName(result.role)}`;
  $("measurements").replaceChildren();
  const foreground=result.regions.reduce((sum,r)=>sum+r.volume_mm3,0);
  const measures=[[tr("影像尺寸","Shape"),dimensions(result.shape,0),tr("体素，按 R / A / S 方向","voxels along R / A / S")],[tr("体素间距","Spacing"),dimensions(result.spacing,3),tr("mm，相邻采样位置的距离","mm between adjacent samples")],[tr("物理覆盖","Coverage"),dimensions(result.coverage_mm,1),tr("mm，当前影像的空间范围","mm of image extent")],[tr("标注总体积","Mask volume"),result.has_mask?format(foreground)+" mm³":tr("未提供","Not provided"),tr("所有非背景标签的体积之和","All non-background labels")]];
  for(const [name,value,hint] of measures){const group=document.createElement("div"),dt=document.createElement("dt"),dd=document.createElement("dd"),small=document.createElement("small");dt.textContent=name;dd.textContent=value;small.textContent=hint;dd.append(small);group.append(dt,dd);$("measurements").append(group);}
  $("window-low").value=result.window[0];$("window-high").value=result.window[1];$("window-preset").value="auto";
  for(const preset of $("window-preset").options){preset.disabled=["lung","soft","bone"].includes(preset.value)&&result.modality.toUpperCase()!=="CT";}
  $("region").replaceChildren(option(-1,tr("全部标注","All regions")));result.regions.forEach(r=>$("region").add(option(r.value,r.name)));
  $("overlay").disabled=!result.has_mask;$("center").disabled=!result.regions.some(r=>r.voxels);$("zoom").value="1";
  if(saved)for(const [id,value] of Object.entries(saved.controls))$(id).value=value;
  updateWindowEditing();
  for(const axis of [0,1,2]){$(`slice-${axis}`).max=result.shape[axis]-1;$(`slice-${axis}`).value=state.cursor[axis];}
  $("geometry-note").textContent=tr(`显示方向 ${result.orientation.join(" / ")}；原始方向 ${result.native_orientation.join(" / ")}。`,`Display axes ${result.orientation.join(" / ")}; native axes ${result.native_orientation.join(" / ")}. `)+(Math.max(...result.obliquity_degrees)>.1 ? tr("倾斜采集：沿原生体素方向显示，解剖方向为近似。","Oblique acquisition: views follow native voxel axes; anatomical directions are approximate.") : "");
  populateDetails(result);drawHistogram(result);
  $("loading").textContent=tr("生成切片与三维视图…","Rendering slices and 3D mask…");
  await Promise.all([refreshSlices(),loadMesh(token)]);
  if(token===state.token)$("loading").textContent="";
}

function populateDetails(d){
  $("region-stats").replaceChildren();
  if(!d.has_mask)$("region-stats").textContent=tr("该病例未提供 mask。","No mask provided for this case.");
  for(const r of d.regions){
    const row=document.createElement("div");row.className="region-row";const swatch=document.createElement("span");swatch.className="swatch";swatch.style.background=r.color;
    const name=document.createElement("span");name.textContent=`${r.value} · ${r.name}`;const details=document.createElement("small");details.textContent=`${format(r.voxels,0)} ${tr("个体素","voxels")}${r.touches_edge?tr(" · 接触影像边界"," · Touches image boundary"):""}`;name.append(details);
    const volume=document.createElement("strong");volume.textContent=`${format(r.volume_mm3)} mm³`;row.append(swatch,name,volume);$("region-stats").append(row);
  }
  const notes={Dataset005_LUNA25:tr("FLARE 提供的结节 mask；其独立人工来源仍待核实。","FLARE nodule masks; independent manual provenance remains unverified."),Dataset003_UCSD_PTGB:tr("部分病例为全背景 mask；临床 IDH 空白与分类标签的对应存在已知来源问题。","Some masks contain only background. The source has a known issue relating blank clinical IDH fields to class labels."),Dataset061_PETWB_Lung:tr("肺器官 mask，来源说明为 TotalSegmentator 标注。","Lung organ masks, described by the source as TotalSegmentator annotations."),Dataset062_PETWB_Liver:tr("肝脏器官 mask，来源说明为 TotalSegmentator 标注。","Liver organ masks, described by the source as TotalSegmentator annotations."),Dataset004_PICAI:tr("前列腺器官与肿瘤标注具有不同来源。","Prostate and tumor annotations have different sources.")};
  $("annotation-note").textContent=notes[d.dataset]??tr("区域名称来自数据集描述；体积根据提供的 mask 计算。","Region names follow the dataset description. Volumes are calculated from supplied masks.");
  const unit=d.modality.toUpperCase()==="CT"?" HU":tr("（原始强度）"," (file intensities)");
  $("intensity-note").textContent=tr(`原始范围 ${format(d.range[0])} 至 ${format(d.range[1])}${unit}；中位数 ${format(d.quantiles[1])}。`,`Range ${format(d.range[0])} to ${format(d.range[1])}${unit}; median ${format(d.quantiles[1])}.`);
  const metadata=[[tr("分类标签","Class label"),labelsText(d.labels)],[tr("来源目录","Source directory"),`images${d.partition}`],[tr("来源 validation fold","Source validation fold"),d.fold??tr("不适用","N/A")],[tr("原始数组尺寸","Native shape"),dimensions(d.native_shape,0)],[tr("原始方向","Native orientation"),d.native_orientation.join(" / ")],[tr("影像压缩文件","Compressed image size"),format(d.image_bytes/1048576)+" MiB"]];
  for(const [key,title] of [["PatientID",tr("患者编号","Patient ID")],["NoduleID",tr("结节编号","Nodule ID")],["StudyDate",tr("检查日期","Study date")],["Age_at_StudyDate",tr("检查时年龄","Age at study")],["Gender",tr("性别","Sex")]]){if(key in d.clinical)metadata.push([title,d.clinical[key]||tr("缺失","Missing")]);}
  $("metadata").replaceChildren();for(const [name,value] of metadata){const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=name;dd.textContent=value;$("metadata").append(dt,dd);}
}
function drawHistogram(d){const counts=d.histogram.counts,edges=d.histogram.edges;Plotly.react("histogram",[{type:"bar",x:counts.map((_,i)=>(edges[i]+edges[i+1])/2),y:counts,width:counts.map((_,i)=>edges[i+1]-edges[i]),marker:{color:"#388670"},hovertemplate:tr("强度 %{x:.1f}<br>体素数量 %{y}<extra></extra>","Intensity %{x:.1f}<br>Voxels %{y}<extra></extra>")}],{margin:{l:55,r:5,t:5,b:25},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:"#596d65",size:10},xaxis:{showgrid:false},yaxis:{title:{text:tr("体素数","Voxels"),font:{size:10}},gridcolor:"#e1e6e1"},bargap:0},{displayModeBar:false,responsive:true});}

async function refreshSlices(){await Promise.all([0,1,2].map(refreshSlice));}
async function refreshSlice(axis){
  if(!state.detail)return;
  state.controllers[axis]?.abort();const controller=new AbortController();state.controllers[axis]=controller;
  const token=state.token,d=state.detail;state.cursor[axis]=Number($(`slice-${axis}`).value);
  $(`position-${axis}`).textContent=`${state.cursor[axis]+1} / ${d.shape[axis]} · ${format(state.cursor[axis]*d.spacing[axis],1)} mm`;
  const params=new URLSearchParams({axis,index:state.cursor[axis],low:$("window-low").value,high:$("window-high").value,overlay:$("overlay").checked,region:$("region").value});
  const response=await fetch(route("slice")+"&"+params,{signal:controller.signal});
  if(!response.ok)throw new Error(await response.text());
  const bitmap=await createImageBitmap(await response.blob());
  if(token!==state.token||controller.signal.aborted){bitmap.close();return;}
  state.images[axis]?.close();state.images[axis]=bitmap;drawPlane(axis);
}
function drawPlane(axis){
  if(!state.detail||!state.images[axis])return;
  const canvas=$(`canvas-${axis}`),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);const context=canvas.getContext("2d");context.scale(ratio,ratio);context.fillStyle="#0b151a";context.fillRect(0,0,rect.width,rect.height);context.imageSmoothingEnabled=false;
  const d=state.detail,[h,v]=[0,1,2].filter(i=>i!==axis),width=d.shape[h]*d.spacing[h],height=d.shape[v]*d.spacing[v];
  const scale=Math.min((rect.width-42)/width,(rect.height-30)/height)*Number($("zoom").value);const w=width*scale,ht=height*scale,x=(rect.width-w)/2,y=(rect.height-ht)/2;
  context.drawImage(state.images[axis],x,y,w,ht);state.rectangles[axis]={x,y,w,h:ht,horizontal:h,vertical:v};
  context.strokeStyle="rgba(103,203,178,.52)";context.lineWidth=.75;context.setLineDash([4,4]);
  const cx=x+(state.cursor[h]+.5)/d.shape[h]*w,cy=y+(1-(state.cursor[v]+.5)/d.shape[v])*ht;
  context.beginPath();context.moveTo(cx,Math.max(y,0));context.lineTo(cx,Math.min(y+ht,rect.height));context.moveTo(Math.max(x,0),cy);context.lineTo(Math.min(x+w,rect.width),cy);context.stroke();context.setLineDash([]);
  context.fillStyle="#d5e8e8";context.font="11px Segoe UI";context.textAlign="center";context.fillText("RAS"[v],rect.width/2,13);context.fillText(({R:"L",A:"P",S:"I"})["RAS"[v]],rect.width/2,rect.height-5);context.fillText("RAS"[h],rect.width-12,rect.height/2);context.fillText(({R:"L",A:"P",S:"I"})["RAS"[h]],12,rect.height/2);
  const mm=width<120?10:50,bar=Math.min(mm*scale,rect.width-60);context.fillStyle="#0b151add";context.fillRect(18,rect.height-39,bar+12,28);context.fillStyle="#fff";context.fillRect(23,rect.height-17,bar,2);context.textAlign="left";context.fillText(`${format(bar/scale,0)} mm`,23,rect.height-23);
}
async function loadMesh(token=state.token){
  clearMesh();const meshToken=state.meshToken;const controller=new AbortController();state.controllers.mesh=controller;
  const pending=document.createElement("p");pending.className="mesh-message";pending.textContent=tr("生成三维标注…","Generating 3D mask…");$("mesh").append(pending);$("mesh-note").textContent=tr("读取区域…","Loading region…");
  const result=await api(route("mesh")+`&region=${$("region").value}`,controller.signal);if(token!==state.token||meshToken!==state.meshToken)return;
  $("mesh").replaceChildren();
  if(!result.meshes.length){const message=document.createElement("p");message.className="mesh-message";message.textContent=state.detail.has_mask?tr("当前区域没有非背景体素。","No foreground voxels in the selected region."):tr("该病例未提供分割标注。","No segmentation mask provided.");$("mesh").append(message);$("mesh-note").textContent="";return;}
  const traces=result.meshes.map(mesh=>({type:"mesh3d",x:mesh.vertices.map(v=>v[0]),y:mesh.vertices.map(v=>v[1]),z:mesh.vertices.map(v=>v[2]),i:mesh.faces.map(f=>f[0]),j:mesh.faces.map(f=>f[1]),k:mesh.faces.map(f=>f[2]),name:mesh.name,color:mesh.color,opacity:result.meshes.length>1?.7:1,flatshading:true,hovertemplate:mesh.name+"<br>R %{x:.1f} mm<br>A %{y:.1f} mm<br>S %{z:.1f} mm<extra></extra>",lighting:{ambient:.65,diffuse:.8,specular:.15}}));
  const axis=title=>({title:{text:title,font:{size:10}},color:"#a9c1c8",backgroundcolor:"#0b151a",gridcolor:"#29404a",zerolinecolor:"#47636c",showbackground:true});
  const plot=document.createElement("div");plot.style.width="100%";plot.style.height="100%";$("mesh").append(plot);state.meshElement=plot;
  await Plotly.newPlot(plot,traces,{paper_bgcolor:"#0b151a",margin:{l:0,r:0,t:0,b:0},scene:{aspectmode:"data",xaxis:axis("R · mm"),yaxis:axis("A · mm"),zaxis:axis("S · mm"),camera:{eye:{x:1.5,y:1.5,z:1.1}}},showlegend:false},{responsive:true,displayModeBar:false});
  if(token!==state.token||meshToken!==state.meshToken)return;state.meshReady=true;
  const step=Math.max(...result.meshes.map(m=>m.step));$("mesh-note").textContent=tr(`拖动旋转 · 滚轮缩放 · ${step===1?"原生表面":`采样步长 ${step} 体素`}`,`Drag to rotate · Scroll to zoom · ${step===1?"Native surface":`Surface step ${step} voxels`}`);
}

$("dataset").onchange=()=>populateDataset();
window.addEventListener('viewer-language-change',()=>{
  const saved=state.detail ? {cursor:[...state.cursor],controls:Object.fromEntries(["window-low","window-high","window-preset","region","zoom"].map(id=>[id,$(id).value]))} : null;
  language=ViewerLanguage.value;translateStatic();
  if(!state.catalog)return;
  populateDataset(true);
  if(state.case)loadCase(state.case,saved).catch(showError);else clearView();
});
let searchTimer;$("search").oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.offset=0;loadList().catch(showError);},220);};
for(const id of ["label-filter","role-filter"]){$(id).onchange=()=>{state.offset=0;updateScope();loadList().catch(showError);};}
$("previous").onclick=()=>{state.offset=Math.max(0,state.offset-60);loadList().catch(showError);};$("next").onclick=()=>{state.offset+=60;loadList().catch(showError);};
$("help-toggle").onclick=()=>{$("help").hidden=!$("help").hidden;};
$("channel").onchange=()=>{if(state.case)loadCase(state.case).catch(showError);};
$("overlay").onchange=()=>refreshSlices().catch(showError);
$("region").onchange=()=>{stopPlay();Promise.all([refreshSlices(),loadMesh()]).catch(showError);};
$("zoom").onchange=()=>[0,1,2].forEach(drawPlane);
$("window-preset").onchange=()=>{updateWindowEditing();if(!state.detail)return;const d=state.detail;const presets={auto:d.window,lung:[-1000,400],soft:[-160,240],bone:[-500,1500],full:d.range};const values=presets[$("window-preset").value];if(values){$("window-low").value=values[0];$("window-high").value=values[1]>values[0]?values[1]:values[0]+1;$("error").hidden=true;refreshSlices().catch(showError);}};
for(const id of ["window-low","window-high"]){$(id).onchange=()=>{$("window-preset").value="custom";if(Number($("window-low").value)>=Number($("window-high").value)){showError(new Error(tr("显示窗下限必须小于上限","Window lower bound must be below upper bound")));return;}$("error").hidden=true;refreshSlices().catch(showError);};}
$("center").onclick=()=>{const selected=state.detail.regions.find(r=>r.value===Number($("region").value));state.cursor=selected?.bounds ? selected.bounds.map(([a,b])=>Math.floor((a+b)/2)) : [...state.detail.center];[0,1,2].forEach(axis=>$(`slice-${axis}`).value=state.cursor[axis]);refreshSlices().catch(showError);};
$("reset-camera").onclick=()=>{if(state.meshReady)Plotly.relayout(state.meshElement,{"scene.camera":{eye:{x:1.5,y:1.5,z:1.1}}});};
for(const axis of [0,1,2]){
  $(`slice-${axis}`).oninput=()=>{refreshSlice(axis).then(()=>[0,1,2].forEach(drawPlane)).catch(showError);};
  const canvas=$(`canvas-${axis}`);
  function moveSlice(delta){if(!state.detail)return;const slider=$(`slice-${axis}`);slider.value=Math.max(0,Math.min(state.detail.shape[axis]-1,Number(slider.value)+delta));slider.dispatchEvent(new Event("input"));}
  canvas.addEventListener("wheel",event=>{event.preventDefault();moveSlice(Math.sign(event.deltaY));},{passive:false});
  canvas.onkeydown=event=>{if(["ArrowUp","ArrowRight","ArrowDown","ArrowLeft"].includes(event.key)){event.preventDefault();moveSlice(["ArrowUp","ArrowRight"].includes(event.key)?1:-1);}};
  canvas.onclick=event=>{if(!state.detail||!state.rectangles[axis])return;const bounds=canvas.getBoundingClientRect(),r=state.rectangles[axis];const x=event.clientX-bounds.left,y=event.clientY-bounds.top;if(x<r.x||x>r.x+r.w||y<r.y||y>r.y+r.h)return;state.cursor[r.horizontal]=Math.min(state.detail.shape[r.horizontal]-1,Math.floor((x-r.x)/r.w*state.detail.shape[r.horizontal]));state.cursor[r.vertical]=Math.min(state.detail.shape[r.vertical]-1,Math.floor((1-(y-r.y)/r.h)*state.detail.shape[r.vertical]));[0,1,2].forEach(a=>$(`slice-${a}`).value=state.cursor[a]);refreshSlices().catch(showError);};
  new ResizeObserver(()=>drawPlane(axis)).observe(canvas.parentElement);
}
$("play").onclick=()=>{if(state.playing){stopPlay();return;}if(!state.detail)return;state.playing=true;$("play").textContent=tr("暂停播放","Pause");const token=state.token;async function tick(){if(!state.playing||token!==state.token)return;$("slice-2").value=(Number($("slice-2").value)+1)%state.detail.shape[2];await refreshSlice(2);[0,1].forEach(drawPlane);setTimeout(()=>tick().catch(error=>{stopPlay();showError(error);}),120);}tick().catch(error=>{stopPlay();showError(error);});};
async function initialize(){state.catalog=await api("/api/catalog");$("dataset").replaceChildren(...state.catalog.datasets.map(d=>option(d.id,`${d.name} · ${format(d.count,0)}`)));$("dataset").value="Dataset005_LUNA25";$("revision").textContent=`· ${state.catalog.revision.slice(0,8)}`;populateDataset();}
initialize().catch(showError);
