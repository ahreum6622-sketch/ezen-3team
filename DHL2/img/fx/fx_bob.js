/* 배 흔들림 — United Carriers와 같은 값
   0초 → 2.4초: 오른쪽 0.6%(배 폭 기준) · 앞으로 1.2%(배 길이 기준)
   2.4초 → 5.1초: 왼쪽 0.45% · 뒤로 2%
   그다음 같은 길을 거꾸로 되돌아옴(왕복), 매 구간 sine.inOut(천천히 출발·천천히 멈춤), 무한 반복 */
(function(){
  const ship=document.getElementById('lgShip'),ct=document.getElementById('lgContTop');
  const ease=t=>-(Math.cos(Math.PI*t)-1)/2;                       // sine.inOut
  const K=[[0,0,0,0],[2.4,.6,-1.2,0],[5.1,-.45,2,.6]];             // [시간, x%, y%, wakeForce]
  const T=5.1;
  function at(sec){
    let u=sec%(2*T); if(u>T) u=2*T-u;                               // 왕복(yoyo)
    const i=u<=2.4?0:1, a=K[i], b=K[i+1], p=ease((u-a[0])/(b[0]-a[0]));
    return [a[1]+(b[1]-a[1])*p, a[2]+(b[2]-a[2])*p, a[3]+(b[3]-a[3])*p];
  }
  window.SHIPBOB=window.SHIPBOB||{on:false};
  const t0=performance.now();
  function loop(now){
    requestAnimationFrame(loop);
    const B=window.SHIPBOB;
    if(!B.on||!B.ppu){ if(ship.style.translate){ship.style.translate='';ct.style.translate='';} return; }
    const sec=(B.tOverride!=null?B.tOverride:(now-t0)/1000);
    const [xp,yp,wf]=at(sec), amp=B.amp||0;
    // UC 배도 우리 배처럼 뱃머리가 화면 아래 → UC와 같은 방향 그대로 사용
    const dx=xp/100*B.w*B.ppu*amp, dy=yp/100*B.h*B.ppu*amp;
    const v=dx.toFixed(2)+'px '+dy.toFixed(2)+'px';
    ship.style.translate=v; ct.style.translate=v;
    B.x=xp*amp; B.y=yp*amp; B.dx=dx; B.dy=dy;
    B.wakeForce=wf;                                                  // UC: 물살 세기 = 0.4 + 0.045 × wakeForce (물살 단계에서 사용)
  }
  requestAnimationFrame(loop);
})();
