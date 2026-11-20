/* DHL 실시간 바다 v9 (물살 시뮬레이션 추가)
   - 물살: UC WakeSimulation과 같은 방식 — 256×256 높이장(파동 방정식 + 물 흐름) 을 GPU에서 매 1/60초 계산
     배 = 움직이는 힘(선체 옆 + 뱃머리 강하게 + 프로펠러), 뱃머리·선미에서 폭이 좁아짐, 결과를 바다 노멀에 섞음
   (아래는 v4 설명)
   DHL 실시간 바다 v4 — United Carriers 바다와 같은 계산 방식
   - 위에서 내려다보는 원근 카메라(FOV 35°) + 물 재질(거칠기 0, 금속감 0.61)
   - 물결 노멀맵 2겹: 크기 1배·2배, 서로 다른 방향으로 흐름 (밉맵 없이 → 멀어지면 잔물결이 반짝이며 움직임)
   - 하늘 환경맵 반사 + 햇빛 반사(강한 반짝임은 압축) + 색 얼룩 + 채도 오버레이 + 파란 빛 번짐 3개 + ACES
   - 크기 기준: 배 길이 126.5m = UC 배 길이 11단위 → 1단위 ≈ 11.5m (배 대비 물결 크기를 UC와 같게) */
(function(){
  const sea=document.getElementById('lgSea'); if(!sea||!window.FX_TEX4) return;
  const cv=document.createElement('canvas'); cv.id='seaGL';
  cv.style.cssText='position:absolute;left:0;top:0;width:100%;height:100%;display:block;pointer-events:none';
  sea.insertBefore(cv,sea.firstChild);
  const gl=cv.getContext('webgl2',{alpha:false,antialias:false,depth:false,stencil:false,powerPreference:'high-performance'});
  const S=window.FXS={on:false,ok:!!gl}; if(!gl) return;

  // simplex noise: Ian McEwan, Stefan Gustavson (webgl-noise, MIT)
  const SNOISE=`vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x,289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){const vec2 C=vec2(1.0/6.0,1.0/3.0);const vec4 D=vec4(0.0,0.5,1.0,2.0);
vec3 i=floor(v+dot(v,C.yyy));vec3 x0=v-i+dot(i,C.xxx);vec3 g=step(x0.yzx,x0.xyz);vec3 l=1.0-g;
vec3 i1=min(g.xyz,l.zxy);vec3 i2=max(g.xyz,l.zxy);vec3 x1=x0-i1+C.xxx;vec3 x2=x0-i2+2.0*C.xxx;vec3 x3=x0-1.+3.0*C.xxx;
i=mod(i,289.0);vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
float n_=1.0/7.0;vec3 ns=n_*D.wyz-D.xzx;vec4 j=p-49.0*floor(p*ns.z*ns.z);vec4 x_=floor(j*ns.z);vec4 y_=floor(j-7.0*x_);
vec4 x=x_*ns.x+ns.yyyy;vec4 y=y_*ns.x+ns.yyyy;vec4 h=1.0-abs(x)-abs(y);vec4 b0=vec4(x.xy,y.xy);vec4 b1=vec4(x.zw,y.zw);
vec4 s0=floor(b0)*2.0+1.0;vec4 s1=floor(b1)*2.0+1.0;vec4 sh=-step(h,vec4(0.0));vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
vec3 p0=vec3(a0.xy,h.x);vec3 p1=vec3(a0.zw,h.y);vec3 p2=vec3(a1.xy,h.z);vec3 p3=vec3(a1.zw,h.w);
vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);m=m*m;
return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));}`;
  const VS=`#version 300 es
in vec2 p;void main(){gl_Position=vec4(p,0.,1.);}`;
  const FS=`#version 300 es
precision highp float;
uniform sampler2D tN,tE,tO,tH;uniform vec2 uShipC;uniform float uShipL;uniform float uWU,uWOff,uWI,uWOn;uniform vec3 uGF,uOF;uniform float uLight,uShadow,uDark;
uniform vec2 uRes,uO; uniform float uPPU,uT,uPx,uZ;
out vec4 o;
const float U=0.02124, MPU=10.;              // 월드1단위(m) / UC 1단위(m)
const float PI=3.14159265, METAL=.61, ROUGH=.0525, NSCALE=.7, OSPD=.9;
vec3 s2l(vec3 c){return mix(c/12.92,pow((max(c,0.)+.055)/1.055,vec3(2.4)),step(vec3(.04045),c));}
vec3 l2s(vec3 c){return mix(c*12.92,1.055*pow(max(c,0.),vec3(1./2.4))-.055,step(vec3(.0031308),c));}
float h21(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float vn(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(h21(i),h21(i+vec2(1,0)),f.x),mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x),f.y);}
float fbm(vec2 p){float a=.5,s=0.;for(int i=0;i<4;i++){s+=a*vn(p);p=p*2.03+vec2(17.1,9.7);a*=.5;}return s;}
// 채도 섞기 (CSS saturation 블렌드와 같은 계산)
float bL(vec3 c){return dot(c,vec3(.3,.59,.11));}
float bS(vec3 c){return max(max(c.r,c.g),c.b)-min(min(c.r,c.g),c.b);}
vec3 clipC(vec3 c){float l=bL(c),n=min(min(c.r,c.g),c.b),x=max(max(c.r,c.g),c.b);
  if(n<0.)c=l+((c-l)*l)/max(l-n,1e-5); if(x>1.)c=l+((c-l)*(1.-l))/max(x-l,1e-5); return c;}
vec3 setL(vec3 c,float l){return clipC(c+(l-bL(c)));}
vec3 setS(vec3 c,float s){float n=min(min(c.r,c.g),c.b),x=max(max(c.r,c.g),c.b);if(x<=n)return vec3(0);return (c-n)*s/(x-n);}
vec3 satBlend(vec3 b,vec3 s){return setL(setS(b,bS(s)),bL(b));}
float radial(vec2 uv,vec2 c){vec2 m=vec2(uRes.x/uRes.y,1.);vec2 p=uv*m,q=c*m;
  float fc=max(max(distance(q,vec2(0)),distance(q,vec2(m.x,0))),max(distance(q,vec2(0,1)),distance(q,vec2(m.x,1))));
  return clamp(1.-distance(p,q)/max(fc*.35,1e-4),0.,1.)*.15;}
// ACES Filmic (three.js와 같은 식)
vec3 RRT(vec3 v){vec3 a=v*(v+.0245786)-.000090537;vec3 b=v*(.983729*v+.4329510)+.238081;return a/b;}
vec3 aces(vec3 c){c*=1./.6;
  const mat3 I=mat3(vec3(.59719,.07600,.02840),vec3(.35458,.90834,.13383),vec3(.04823,.01566,.83777));
  const mat3 O=mat3(vec3(1.60475,-.10208,-.00327),vec3(-.53108,1.10813,-.07276),vec3(-.07367,-.00605,1.07602));
  return clamp(O*RRT(I*c),0.,1.);}
// 환경맵(하늘) 읽기: 화면좌표계(x 오른쪽, y 아래, z 위) → 하늘 사진 좌표
vec3 env(vec3 d){d.z=max(d.z,.03);d=normalize(d);vec3 w=vec3(d.x,d.z,d.y);float u=atan(w.z,w.x)/(2.*PI)+.5+1.47/(2.*PI),v=asin(clamp(w.y,-1.,1.))/PI+.5;
  return s2l(textureLod(tE,vec2(u,1.-v),0.).rgb);}
${SNOISE}
vec2 rot(vec2 v,float a){float c=cos(a),s=sin(a);return vec2(c*v.x-s*v.y,s*v.x+c*v.y);}
void main(){
  vec2 fc=vec2(gl_FragCoord.x,uRes.y-gl_FragCoord.y);
  vec2 su=fc/uRes;                                   // 화면 uv (위=0)
  vec2 uu=(fc-uO)/uPPU*U/MPU;                        // 이 픽셀의 바다 위치 (UC 단위)
  vec2 cu=(uRes*.5-uO)/uPPU*U/MPU;                   // 화면 중앙
  float visH=uRes.y*uPx/MPU;                         // 화면 높이 (UC 단위)
  vec3 cam=vec3(cu,visH*.5/tan(radians(17.5)));      // 카메라 높이 (FOV 35°)
  vec3 P=vec3(uu,0.),V=normalize(cam-P);
  // ---- 노멀맵 2겹 (UC: 2000 평면에 68번 반복, 1.2rad 회전) ----
  // ---- 물살 좌표: 배 중심 기준 (UC 단위: 배 길이 = 11), y는 화면 위쪽이 + ----
  vec2 pw=(fc-uO)/uPPU, rp=(pw-uShipC)/uWU, pl=vec2(rp.x,-rp.y);
  vec2 wuv=vec2(.5+pl.x/60.,.5+(pl.y-uWOff)/60.);
  float e08=smoothstep(0.,.08,wuv.x)*smoothstep(0.,.08,1.-wuv.x)*smoothstep(0.,.08,wuv.y)*smoothstep(0.,.08,1.-wuv.y);
  float e25=smoothstep(0.,.25,wuv.x)*smoothstep(0.,.25,1.-wuv.x)*smoothstep(0.,.25,wuv.y)*smoothstep(0.,.25,1.-wuv.y);
  vec2 wd=vec2(pl.x,-pl.y-uWOff);                     // UC wakeDiff
  float dep=max(0.,-wd.y), wf=2.5+dep*.25, nx=wd.x/wf;
  float fo=smoothstep(60.,0.,dep)*smoothstep(-6.,3.,wd.y), ps=2.2*uWI;
  vec2 uvOff=vec2(-nx*exp(-nx*nx*2.)*ps*fo,-exp(-nx*nx*2.)*ps*.5*fo)*.002*uWOn;   // 배가 물을 가르며 무늬가 양옆으로 밀림
  vec2 q=rot(vec2(uu.x,-uu.y)/(2000./68.),1.2);
  vec3 n1=texture(tN,q+vec2(-.02,-.01)*uT*OSPD+uvOff).xyz*2.-1.; n1.xy*=NSCALE;
  vec3 n2=texture(tN,q*2.+vec2(.01,-.015)*uT*OSPD+uvOff*2.).xyz*2.-1.; n2.xy=-n2.xy; n2.xy*=NSCALE;
  vec3 ocn=vec3(n1.xy+n2.xy,n1.z*n2.z);
  // 배 뒤 V자 구역: 주변 잔물결 85% 잠재움 (물살이 잘 보이게)
  vec2 pd=vec2(wd.x,-wd.y); float dd=max(0.,pd.y), hw=2.2+dd*.28;
  float dmp=smoothstep(hw,hw*.35,abs(pd.x))*smoothstep(24.,0.,dd)*smoothstep(-1.5,0.,pd.y)*e08;
  ocn.xy*=1.-dmp*.85*uWOn;
  // 물살 높이장 → 노멀 (UC: 셀 3.2칸 간격, 세기 2.5 × 높이배율)
  float zp=clamp(log(uZ/.8)/log(.048/.8),0.,1.), hs=2.5-.8*mix(1.,.082,zp);
  vec2 cl=vec2(3.2/256.); vec2 wc=clamp(wuv,0.,1.);
  float hL=texture(tH,wc-vec2(cl.x,0)).x,hR=texture(tH,wc+vec2(cl.x,0)).x,hU=texture(tH,wc+vec2(0,cl.y)).x,hD=texture(tH,wc-vec2(0,cl.y)).x;
  vec2 wn=vec2((hL-hR)*2.5*hs,(hD-hU)*2.5*hs)*e25*1.1;
  vec3 rq=vec3(rot(ocn.xy,-1.2)+wn,ocn.z);
  vec3 N=normalize(vec3(rq.x,-rq.y,rq.z));
  // ---- 바탕색: 위·아래 색 사이 + 느린 얼룩 ----
  float fn=(fbm(uu*.011*1.1*8.+uT*.5*vec2(.13,.07))-.5)*2.;
  vec3 base=mix(s2l(vec3(25,54,83)/255.),s2l(vec3(2,20,54)/255.),.5+fn*.05);
  // ================= 거품 (UC와 같은 계산) =================
  float ff=0., vF=0.;
  vec2 fgo=vec2(.5,.5+uWOff/60.); float fgd=wuv.y-fgo.y, fgw=.07+max(fgd,0.)*.36;
  bool inW=fgd>=-.15&&fgd<=.4&&abs(wuv.x-fgo.x)<=fgw;
  vec2 wpos=vec2(pl.x,-pl.y);                                   // UC 월드 x,z (배 기준)
  if(uWOn>0.&&inW){
    vec2 cs2=vec2(1./256.);
    float hV=texture(tH,wuv).x,fL=texture(tH,wuv-vec2(cs2.x,0)).x,fR=texture(tH,wuv+vec2(cs2.x,0)).x,fU=texture(tH,wuv+vec2(0,cs2.y)).x,fD=texture(tH,wuv-vec2(0,cs2.y)).x;
    float crest=max(0.,-((fL+fR+fU+fD)-4.*hV))*38.;               // 파도 꼭대기
    vec2 fuvA=vec2(wpos.x*2.4,wpos.y*.32);                         // 진행 방향으로 늘어난 거품 줄무늬
    float n1f=1.-abs(snoise(vec3(fuvA*1.5,uT*.15))), n2f=1.-abs(snoise(vec3(fuvA*3.5+vec2(0.,uT*.4),uT*.25)));
    float nn=n1f*.6+n2f*.4;
    vec2 fine=vec2(wpos.x*30.,wpos.y*2.);
    float fineF=smoothstep(.4,.75,snoise(vec3(fine+vec2(0.,uT),uT*.5))*.5+.5);
    float dbs=max(0.,wuv.y-.6), dth=.83*clamp(1.-dbs*.95,.4,1.), eth=dth*(1.7-nn*1.1);
    ff=smoothstep(eth,eth+.08,crest)*fineF*nn*2.2*smoothstep(.55,.10,dbs)*.2;
    // 옆 V자 거품
    vec2 p=(wuv-fgo)*16.; float dpt=max(0.,p.y), wfa=.88+dpt*.075, nX=p.x/wfa;
    float vM=exp(-pow(abs(nX)-.40,2.)*12.), fdo=smoothstep(12.,0.,dpt)*smoothstep(-.5,.2,p.y);
    vec2 fv=p*5.8; fv.y-=uT*.3*OSPD; fv.y*=mix(1.,.15,smoothstep(3.,12.,dpt));
    float fnv=.5*(1.-abs(snoise(vec3(fv,uT*.036))))+.25*(1.-abs(snoise(vec3(fv*2.1,-uT*.36))))+.125*(1.-abs(snoise(vec3(fv*4.3,uT*.45))));
    float vB=smoothstep(.55,.85,fnv)*mix(1.,fineF*1.5,smoothstep(0.,6.,dpt));
    float shp=fdo*.7+nn*.5, fA=smoothstep(.55,.9,shp), dis=mix(1.,fA,smoothstep(1.5,9.,dpt));
    vF=vM*vB*dis*fdo*smoothstep(-.5,.2,p.y)*smoothstep(-.05,.25,crest+hV*.5);
    ff=max(ff,vF*.2*1.2);
    // 배 뒤로 넓어지는 세척 거품
    float db2=max(0.,wuv.y-.6);
    if(db2>0.){float whw=.03+db2*.15, we=smoothstep(whw,whw*.25,abs(wuv.x-.5));
      ff=max(ff,we*nn*fineF*smoothstep(.32,.02,db2)*.75*.4*.2);}
    // 뱃머리 바로 앞 화살표 모양 거품 제거
    ff*=1.-(1.-smoothstep(-1.5,-.1,p.y))*(1.-smoothstep(.10,.28,abs(p.x)));
  }
  float hl=(uShipL*60.)*.5, fs0=-hl+uShipL*60.*.15;                 // 뱃머리 = -hl, 시작점 = 뱃머리에서 15% 뒤
  ff*=smoothstep(fs0-1.2,fs0+1.6,pl.y);
  float fF=clamp(ff*e25*1.4,0.,1.);   // v19: UC보다 진하다는 피드백 → 흐릿하게   // UC 화면 실측 밝기에 맞춘 보정 (같은 거품 분포, 밝기만 UC 수준으로)
  vec2 pc=(wuv-fgo)*16.; float dcl=max(0.,pc.y);
  vec3 cFoam=mix(vec3(1.),vec3(.545,.714,.871),clamp(dcl/9.,0.,1.));
  base=mix(base,mix(vec3(.70,.80,.90),cFoam*.85,clamp(vF*2.,0.,1.)),fF);   // v19: 거품 색 조금 어둡게
  float metal=mix(METAL,0.,fF), rough=mix(ROUGH,.97,fF);
  vec3 dif=base*(1.-metal), F0=mix(vec3(.04),base,metal);
  float NV=clamp(dot(N,V),1e-4,1.);
  // 햇빛 (방향광 3.9) + 주변광 0.9
  vec3 L=normalize(vec3(-.0014,.438,.899));
  float NL=clamp(dot(N,L),0.,1.); vec3 Hh=normalize(L+V); float NH=clamp(dot(N,Hh),0.,1.),VH=clamp(dot(V,Hh),0.,1.);
  float a=rough*rough,a2=a*a;
  float D=a2/(PI*pow(NH*NH*(a2-1.)+1.,2.));
  float Vg=.5/max(NL*sqrt(a2+(1.-a2)*NV*NV)+NV*sqrt(a2+(1.-a2)*NL*NL),1e-5);
  vec3 Fs=F0+(1.-F0)*pow(1.-VH,5.);
  vec3 dSpec=3.9*NL*Fs*Vg*D;
  float pk=max(max(dSpec.r,dSpec.g),dSpec.b);           // 강한 반짝임만 압축 (knee .25, 압축 4)
  if(pk>.25){float e=pk-.25;dSpec*=(.25+e/(1.+4.*e))/pk;}
  vec3 dDif=3.9*NL*dif/PI;
  // 환경 반사
  vec4 c0=vec4(-1.,-.0275,-.572,.022),c1=vec4(1.,.0425,1.04,-.04);vec4 r=rough*c0+c1;
  float a004=min(r.x*r.x,exp2(-9.28*NV))*r.x+r.y; vec2 fab=vec2(-1.04,1.04)*a004+r.zw;
  vec3 FssEss=F0*fab.x+fab.y;
  vec3 R=reflect(-V,N);
  vec3 iSpec=mix(env(R),s2l(vec3(.6,.7,.85)),fF)*FssEss;   // 거품은 거칠어서 반사가 뿌옇게
  vec3 irr=.9+s2l(vec3(.36,.54,.83))*PI;                  // 주변광 + 하늘 전체에서 오는 빛
  vec3 iDif=irr*dif/PI*(1.-FssEss);
  // 빛 줄이기: 햇빛 반짝임·하늘 반사를 낮춤
  float wm=clamp(length(wn)*3.,0.,1.)*smoothstep(-(uShipL*30.)+uShipL*60.*.15-1.2,-(uShipL*30.)+uShipL*60.*.15+1.6,pl.y);   // 물살 물결이 있는 곳 (뱃머리 15% 뒤부터)
  vec3 col=dDif+dSpec*uLight+iSpec*uLight+iDif;   // 피드백 반영: UC처럼 밝게 (빛 줄인 것 되돌림)   // 물살 자리도 주변 바다와 같은 밝기 (불 켜진 느낌 제거)
  // ---- 화면 오버레이: 채도 얼룩 + 파란 빛 번짐 3개 ----
  vec2 cs=su; float sa=uRes.x/uRes.y, ta=1920./1288.; vec2 vr=sa>ta?vec2(1.,ta/sa):vec2(sa/ta,1.);
  vec3 ovc=clamp(texture(tO,(cs-.5)*vr+.5).rgb*1.8,0.,1.);
  float ol=dot(ovc,vec3(.213,.715,.072)); ovc=clamp(mix(vec3(ol),ovc,1.2),0.,1.);
  vec3 bd=l2s(clamp(col,0.,1.)); vec3 bl=satBlend(bd,ovc);
  vec3 gc=vec3(0.,160./255.,1.);
  bl=mix(bl,gc,radial(su,vec2(.25,.35)));bl=mix(bl,gc,radial(su,vec2(.75,.35)));bl=mix(bl,gc,radial(su,vec2(.5,.7)));
  col=s2l(clamp(bl,0.,1.))+max(col-1.,0.);
  vec3 oc=l2s(aces(col));
  // UC 화면과 색 맞춤: 가까이(배 큼) / 멀리(배 작음) 각각 UC 캡처 색에 맞춘 보정, 축소 정도에 따라 자연스럽게 이어짐
  float zt=clamp(log(uZ/.8)/log(.048/.8),0.,1.);
  vec3 gN=vec3(1.,1.,.80), gF=uGF, oF=uOF;
  oc=oc*mix(gN,gF,zt)+mix(vec3(0),oF,zt);
  // 그림자 ① 구름 그림자: 크고 부드러운 어두운 얼룩이 천천히 흘러감
  float sh1=fbm(uu*.045+uT*vec2(.010,.006)), sh2=fbm(uu*.016-uT*vec2(.004,.007)+3.7);
  float csh=smoothstep(.40,.72,sh1*.55+sh2*.45);
  // 그림자 ② 물결 골: 해 반대쪽으로 기운 면을 어둡게 → 너울 입체감
  float tr=smoothstep(-.06,.10,dot(N.xy,L.xy));
  float sh=mix(1.,.55,csh*uShadow)*mix(.78,1.,mix(1.,tr,uShadow));
  oc*=sh*uDark;
  o=vec4(clamp(oc,0.,1.),1.);
  o.rgb=mix(o.rgb,clamp(l2s(aces(col)),0.,1.)*.82,smoothstep(0.,.25,fF)*.75);   // v19: 거품을 반투명·조금 어둡게 → 흐릿하게
  // (v16) 제가 임의로 만들었던 배 뒤 물길 띠·그물 거품·배 옆 퍼지는 물보라는 모두 제거 — UC 계산식만 사용
   // 거품은 어둡게·그림자 보정에서 제외 (UC처럼 하얗게)
}`;
  function sh(t,src){const s=gl.createShader(t);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(s));return s}
  const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,VS));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,FS));gl.linkProgram(pr);gl.useProgram(pr);
  if(!gl.getProgramParameter(pr,gl.LINK_STATUS)){console.error(gl.getProgramInfoLog(pr));S.ok=false;return}
  const bf=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,bf);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  const lp=gl.getAttribLocation(pr,'p');gl.enableVertexAttribArray(lp);gl.vertexAttribPointer(lp,2,gl.FLOAT,false,0,0);
  const UN={};['uRes','uO','uPPU','uT','uPx','uZ','uGF','uOF','uLight','uShadow','uDark','uShipC','uShipL','uWU','uWOff','uWI','uWOn','tH'].forEach(n=>UN[n]=gl.getUniformLocation(pr,n));
  function tex(unit,name,url,mip,aniso,fill){
    const t=gl.createTexture();gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(fill));
    gl.uniform1i(gl.getUniformLocation(pr,name),unit);
    const im=new Image();im.onload=()=>{gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,im);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,name==='tE'?gl.CLAMP_TO_EDGE:gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
      if(mip){gl.generateMipmap(gl.TEXTURE_2D);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR)}
      else gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);   // UC처럼 밉맵 없음
      const ext=gl.getExtension('EXT_texture_filter_anisotropic');if(ext&&aniso)gl.texParameterf(gl.TEXTURE_2D,ext.TEXTURE_MAX_ANISOTROPY_EXT,aniso);};
    im.src=url;
  }
  tex(0,'tN',window.FX_TEX4.waterN,1,16,[128,128,255,255]);
  tex(1,'tE',window.FX_TEX4.sky,0,0,[90,130,190,255]);tex(2,'tO',window.FX_TEX4.ov,0,0,[5,40,90,255]);


  // ================= 물살 시뮬레이션 (UC WakeSimulation과 같은 계산) =================

  const SIM_FS=`#version 300 es
precision highp float;
uniform sampler2D heightmap;uniform vec2 resolution,uShipPos,uShipSize;uniform float uWakeIntensity,uTime,uDelta,uShipSpeed,uSmoothing;
out vec4 o;
#define BOUNDS 60.0
#define NOISE_SCALE 1.1
${SNOISE}
void main(){
  vec2 uv=gl_FragCoord.xy/resolution, texel=1.0/resolution;
  float dt=min(uDelta,0.033);
  float uvSpeed=(uShipSpeed/BOUNDS)*0.38;                 // 물이 배 뒤로 흘러감 (UC: 0.38배로 천천히)
  vec2 src=uv-vec2(0.0,uvSpeed*dt);
  vec4 st=texture(heightmap,src);float h=st.r,v=st.g;
  float hL=texture(heightmap,src+vec2(-texel.x,0.)).r,hR=texture(heightmap,src+vec2(texel.x,0.)).r;
  float hD=texture(heightmap,src+vec2(0.,-texel.y)).r,hU=texture(heightmap,src+vec2(0.,texel.y)).r;
  float lap=(hL+hR+hD+hU)-4.0*h;
  v+=lap*0.12; v*=0.95; h*=0.97; h+=v;                    // 파동 방정식 + 감쇠
  vec2 rel=(uv-uShipPos)*BOUNDS;
  float hW=uShipSize.x*0.5*BOUNDS, hLh=uShipSize.y*0.5*BOUNDS;
  vec2 wp=(uv-vec2(0.5))*BOUNDS;
  float amb=snoise(vec3(wp.x*0.01*NOISE_SCALE,wp.y*0.01*NOISE_SCALE,uTime*0.5))*0.15;
  if(rel.y>-hLh&&rel.y<hLh){
    float lhw=hW;
    if(rel.y<-hLh*0.4){float t=(rel.y+hLh)/(hLh*0.6);lhw=hW*smoothstep(0.,1.,t);}      // 뱃머리에서 좁아짐
    else if(rel.y>hLh*0.7){float t=(hLh-rel.y)/(hLh*0.3);lhw=hW*smoothstep(0.,1.,t);}   // 선미에서 좁아짐
    float dside=abs(rel.x)-lhw, dpk=dside-hW*0.08, iw=max(hW*0.18,texel.x*BOUNDS*0.55);
    if(abs(dpk)<iw){
      float f=smoothstep(iw,0.,abs(dpk)), spd=clamp(uShipSpeed/35.0,0.2,1.5);
      float lw=smoothstep(-hLh,-hLh*0.4,rel.y), fs=mix(0.12,0.55,lw)*uWakeIntensity*spd;   // 뱃머리가 더 세게 밀어냄
      float hn=snoise(vec3(wp*1.6,uTime*0.4))*0.25+0.85;
      float force=f*fs*(hn+max(0.,amb)*2.0);
      h=mix(h,force,0.45); v=0.0;
    }
  }
  vec2 pp=vec2(0.,hLh*0.9); float dp=length(rel-pp), pr=hW*0.8;                    // 프로펠러 물보라
  if(dp<pr){float pn=snoise(vec3(rel*8.0,uTime*1.2))*0.15-0.85;v+=pn*uWakeIntensity*0.05*smoothstep(pr,pr*0.2,dp);}
  float sm=(h*4.0+hL+hR+hD+hU)/8.0; h=mix(h,sm,clamp(uSmoothing,0.,1.));
  float e=0.05, fd=smoothstep(0.,e,uv.x)*smoothstep(0.,e,1.-uv.x)*smoothstep(0.,e,uv.y)*smoothstep(0.,e,1.-uv.y);
  h*=fd; v*=fd; h=clamp(h,-2.,2.); v=clamp(v,-2.,2.);
  o=vec4(h,v,0.,1.);
}`;
  const RES=256, cbf=gl.getExtension('EXT_color_buffer_float'), fl=gl.getExtension('OES_texture_float_linear');   // UC처럼 32비트 실수 (16비트면 작은 물결 변화가 사라져 파도가 퍼지지 못함)
  const sp=gl.createProgram();gl.attachShader(sp,sh(gl.VERTEX_SHADER,VS));gl.attachShader(sp,sh(gl.FRAGMENT_SHADER,SIM_FS));gl.linkProgram(sp);
  const simOK=!!cbf&&gl.getProgramParameter(sp,gl.LINK_STATUS); if(!simOK)console.warn('wake sim off',gl.getProgramInfoLog(sp));
  const SU={};['heightmap','resolution','uShipPos','uShipSize','uWakeIntensity','uTime','uDelta','uShipSpeed','uSmoothing'].forEach(n=>SU[n]=gl.getUniformLocation(sp,n));
  const RT=[0,1].map(()=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,RES,RES,0,gl.RGBA,gl.FLOAT,null);
    const FL=fl?gl.LINEAR:gl.NEAREST;gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,FL);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,FL);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    const f=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,f);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,t,0);
    gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);return {t,f}});
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  let cur=0,acc=0,simT=0;
  function simStep(dt,wi,shipW,shipL){
    gl.useProgram(sp);gl.bindFramebuffer(gl.FRAMEBUFFER,RT[1-cur].f);gl.viewport(0,0,RES,RES);
    gl.activeTexture(gl.TEXTURE3);gl.bindTexture(gl.TEXTURE_2D,RT[cur].t);gl.uniform1i(SU.heightmap,3);
    gl.uniform2f(SU.resolution,RES,RES);gl.uniform2f(SU.uShipPos,.5,.5+S.wOff/60);gl.uniform2f(SU.uShipSize,shipW,shipL);
    gl.uniform1f(SU.uWakeIntensity,wi);gl.uniform1f(SU.uTime,simT);gl.uniform1f(SU.uDelta,dt);gl.uniform1f(SU.uShipSpeed,50);gl.uniform1f(SU.uSmoothing,.55);
    gl.drawArrays(gl.TRIANGLES,0,3);cur=1-cur;S.nSim=(S.nSim||0)+1;
  }
  S.readWake=()=>{gl.bindFramebuffer(gl.FRAMEBUFFER,RT[cur].f);const px=new Float32Array(RES*RES*4);gl.readPixels(0,0,RES,RES,gl.RGBA,gl.FLOAT,px);gl.bindFramebuffer(gl.FRAMEBUFFER,null);let mx=0,mn=0;for(let i=0;i<px.length;i+=4){mx=Math.max(mx,px[i]);mn=Math.min(mn,px[i])}return [mn,mx,S.nSim];};
  S.dumpWake=()=>{gl.bindFramebuffer(gl.FRAMEBUFFER,RT[cur].f);const px=new Float32Array(RES*RES*4);gl.readPixels(0,0,RES,RES,gl.RGBA,gl.FLOAT,px);gl.bindFramebuffer(gl.FRAMEBUFFER,null);const o=[];for(let i=0;i<px.length;i+=4)o.push(Math.round(px[i]*1000));return {fl:!!fl,d:o};};
  S.wOff=0.;   // 계산용 배를 배 그림과 정확히 겹침 (UC는 -0.5지만 우리 배는 더 길어서 그림보다 앞으로 튀어나왔음)
  let last=performance.now(),q=1,avg=.016;const t0=performance.now();
  S.tOverride=null;S.lockQ=false;
  function frame(now){
    requestAnimationFrame(frame);
    const dt=Math.min(.05,(now-last)/1000);last=now;
    if(!S.on){S.wasOn=false;return;}
    if(!S.wasOn){S.wasOn=true;if(!S.warmed){S.warm=300;S.warmed=true;}}
    avg+=(dt-avg)*.05;if(!S.lockQ&&avg>.024&&q>.6){q-=.1;avg=.016}
    const dpr=Math.min(devicePixelRatio||1,1.5)*q,cw=Math.round(cv.clientWidth*dpr),ch=Math.round(cv.clientHeight*dpr);
    if(cw<2||ch<2) return;
    if(cv.width!==cw||cv.height!==ch){cv.width=cw;cv.height=ch;gl.viewport(0,0,cw,ch)}
    const ppu=S.ppu*dpr;
    // ---- 물살 계산 (고정 1/60초, 한 프레임 최대 3번) ----
    const B=window.SHIPBOB||{}, wi=.4+.045*(B.wakeForce||0);   // UC처럼 처음부터 물살·물보라 있음 (배가 항상 항해 중인 것처럼)
    const wU=(S.sw||954)/2.2, shipW=2.2/60, shipL=(S.sl||5949)/wU/60;   // 배 폭을 UC 배 폭(2.2)에 맞춤 → 물보라가 UC처럼 선체에 붙음 (배 길이는 우리 배 그대로)
    if(simOK){acc+=dt;let n=0;let MX=S.maxSteps||3;if(S.simBoost){acc+=S.simBoost/60;}
      if(S.warm>0){const w=Math.min(S.warm,30);acc+=w/60;MX+=w;S.warm-=w;}   // 장면에 처음 들어올 때 미리 계산해 둠 (UC 예열과 같은 역할)
      while(acc>=1/60&&n<MX){simT+=1/60;simStep(1/60,wi,shipW,shipL);acc-=1/60;n++}if(n>=MX)acc=Math.min(acc,1/60);
      gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,cw,ch);}
    gl.useProgram(pr);
    gl.activeTexture(gl.TEXTURE3);gl.bindTexture(gl.TEXTURE_2D,RT[cur].t);gl.uniform1i(UN.tH,3);
    gl.uniform2f(UN.uShipC,((S.sx||0)+(S.sw||0)/2)*dpr/dpr,(S.sy||0)+(S.sl||0)/2);gl.uniform1f(UN.uWU,wU);gl.uniform1f(UN.uShipL,shipL);gl.uniform1f(UN.uWOff,S.wOff);gl.uniform1f(UN.uWI,wi);gl.uniform1f(UN.uWOn,simOK?1:0);
    gl.uniform2f(UN.uRes,cw,ch);gl.uniform2f(UN.uO,S.ox*dpr,S.oy*dpr);gl.uniform1f(UN.uPPU,ppu);
    gl.uniform1f(UN.uPx,0.02124/ppu);gl.uniform1f(UN.uZ,S.z||1);const G=S.gF||[1,1.077,.672],O=S.oF||[.054,0,0];gl.uniform3f(UN.uGF,G[0],G[1],G[2]);gl.uniform3f(UN.uOF,O[0],O[1],O[2]);gl.uniform1f(UN.uLight,S.light??1);gl.uniform1f(UN.uShadow,S.shadow??0);gl.uniform1f(UN.uDark,S.dark??1);
    gl.uniform1f(UN.uT,S.tOverride!=null?S.tOverride:(now-t0)/1000);
    gl.drawArrays(gl.TRIANGLES,0,3);
  }
  requestAnimationFrame(frame);
})();
