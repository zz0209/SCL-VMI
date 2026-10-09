'use strict';
class WholeVolumeRenderer {
  constructor() {
    this.canvas=document.createElement('canvas');
    const gl=this.canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true});
    if(!gl)throw new Error('此浏览器无法创建 WebGL2。请启用浏览器图形加速后重新加载。');
    this.gl=gl;this.textures=[];
    const vertex=`#version 300 es
    void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0,1);}`;
    const fragment=`#version 300 es
    precision highp float;
    precision highp sampler3D;
    precision highp isampler2D;
    uniform sampler3D ctData;
    uniform sampler3D mapData;
    uniform isampler2D indices;
    uniform int axis;
    uniform int position;
    uniform vec2 windowRange;
    uniform float scale;
    uniform float opacity;
    uniform float threshold;
    out vec4 outputColor;
    vec3 heat(float t){
      vec3 stops[6]=vec3[6](vec3(21,11,55),vec3(91,28,110),vec3(170,51,91),vec3(229,107,56),vec3(251,182,26),vec3(252,255,164));
      float s=clamp(t,0.0,0.99999)*5.0;int i=int(floor(s));return floor(mix(stops[i],stops[i+1],fract(s))+.5);
    }
    void main(){
      ivec2 q=ivec2(gl_FragCoord.xy);
      ivec3 p=axis==2?ivec3(q.x,q.y,position):axis==1?ivec3(q.x,position,q.y):ivec3(position,q.x,q.y);
      ivec3 m=ivec3(texelFetch(indices,ivec2(p.x,0),0).r,texelFetch(indices,ivec2(p.y,1),0).r,texelFetch(indices,ivec2(p.z,2),0).r);
      float ct=texelFetch(ctData,p.zyx,0).r;
      float value=texelFetch(mapData,m.zyx,0).r;
      float grey=clamp((ct-windowRange.x)/(windowRange.y-windowRange.x)*255.0,0.0,255.0);
      float t=value/scale;
      float a=value>0.0&&t>=threshold?opacity:0.0;
      outputColor=vec4(mix(vec3(grey),heat(t),a)/255.0,1.0);
    }`;
    const compile=(type,source)=>{const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader));return shader;};
    const program=gl.createProgram();gl.attachShader(program,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(program,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);gl.disable(gl.DITHER);this.program=program;this.locations={};
    for(const name of ['ctData','mapData','indices','axis','position','windowRange','scale','opacity','threshold'])this.locations[name]=gl.getUniformLocation(program,name);
    gl.uniform1i(this.locations.ctData,0);gl.uniform1i(this.locations.mapData,1);gl.uniform1i(this.locations.indices,2);
    this.canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();report(new Error('图形加速连接已中断，请重新加载页面。'));});
  }
  upload(meta,ct,activity,mode){
    const gl=this.gl;
    if(ct.length!==meta.shape.reduce((a,b)=>a*b)||activity.length!==meta.map_shape.reduce((a,b)=>a*b))throw new Error('影像数据长度与空间尺寸不一致。');
    this.meta=meta;this.ct=ct;this.activity=activity;
    for(const texture of this.textures)gl.deleteTexture(texture);
    this.textures=[];
    for(const [unit,data,shape] of [[0,ct,meta.shape],[1,activity,meta.map_shape]]){
      gl.activeTexture(gl.TEXTURE0+unit);const texture=gl.createTexture();this.textures.push(texture);gl.bindTexture(gl.TEXTURE_3D,texture);
      gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
      gl.texImage3D(gl.TEXTURE_3D,0,gl.R32F,shape[2],shape[1],shape[0],0,gl.RED,gl.FLOAT,data);
    }
    const width=Math.max(...meta.shape),indices=new Int32Array(width*3);
    this.lookup=meta.shape.map((size,axis)=>{
      const result=new Int32Array(size);
      for(let i=0;i<size;i++){
        const sample=(i*meta.spacing[axis]+meta.origin[axis]-meta.map_origin[axis])/meta.map_spacing_mm;
        const low=Math.floor(sample),fraction=sample-low;
        const index=mode==='spatial'?(fraction===.5?(low%2===0?low:low+1):Math.round(sample)):low;
        result[i]=Math.max(0,Math.min(meta.map_shape[axis]-1,index));indices[axis*width+i]=result[i];
      }
      return result;
    });
    gl.activeTexture(gl.TEXTURE2);const texture=gl.createTexture();this.textures.push(texture);gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.R32I,width,3,0,gl.RED_INTEGER,gl.INT,indices);
    if(gl.getError()!==gl.NO_ERROR)throw new Error('影像未能完整传入图形加速器，请重新加载页面。');
  }
  point(point){
    const shape=this.meta.shape,mapShape=this.meta.map_shape,m=point.map((value,axis)=>this.lookup[axis][value]);
    return [this.ct[(point[0]*shape[1]+point[1])*shape[2]+point[2]],this.activity[(m[0]*mapShape[1]+m[1])*mapShape[2]+m[2]]];
  }
  render(axis,position,settings){
    const axes=[0,1,2].filter(value=>value!==axis),[nx,ny]=axes.map(value=>this.meta.shape[value]),gl=this.gl;
    if(this.canvas.width!==nx||this.canvas.height!==ny){this.canvas.width=nx;this.canvas.height=ny;}
    gl.viewport(0,0,nx,ny);gl.uniform1i(this.locations.axis,axis);gl.uniform1i(this.locations.position,position);
    gl.uniform2fv(this.locations.windowRange,settings.window);gl.uniform1f(this.locations.scale,settings.scale);gl.uniform1f(this.locations.opacity,settings.opacity);gl.uniform1f(this.locations.threshold,settings.threshold);
    gl.drawArrays(gl.TRIANGLES,0,3);
    return this.canvas;
  }
}
