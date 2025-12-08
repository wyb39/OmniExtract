(()=>{
  function init(){
    const el = document.getElementById('realtime-log-content-container');
    if(!el) return;
    try {
      const obs = new MutationObserver(()=>{
        try{
          el.scrollTo({top: el.scrollHeight, behavior: 'smooth'});
        }catch(e){
          el.scrollTop = el.scrollHeight;
        }
      });
      obs.observe(el, {childList: true, subtree: true});
    } catch (e) {
      // no-op
    }
  }
  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();