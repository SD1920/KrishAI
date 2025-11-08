// frontend/app.js
const BACKEND = "http://127.0.0.1:8000";

function $(id){ return document.getElementById(id); }

function toHa(value, unit){
  const v = parseFloat(value) || 0;
  if(unit === "ha") return v;
  if(unit === "acre") return v * 0.40468564224;
  if(unit === "guntha") return v * 0.025;
  if(unit === "m2") return v / 10000;
  return v;
}

async function reverseGeocode(lat, lon){
  try{
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}&zoom=10`;
    const res = await fetch(url);
    if(!res.ok) return null;
    const j = await res.json();
    const addr = j.address || {};
    const prefer = ["county","state_district","city","town","village","municipality","state","region"];
    for(const k of prefer){
      if(addr[k]) return addr[k];
    }
    if(j.display_name) return j.display_name.split(",")[0];
    return null;
  }catch(e){
    return null;
  }
}

function showBusy(on){
  const btns = document.querySelectorAll("button");
  btns.forEach(b => b.disabled = on);
}

function showToast(msg){
  try {
    const el = document.createElement("div");
    el.textContent = msg;
    el.style.position = "fixed";
    el.style.right = "18px";
    el.style.bottom = "18px";
    el.style.padding = "10px 14px";
    el.style.background = "#10b981";
    el.style.color = "white";
    el.style.borderRadius = "8px";
    el.style.boxShadow = "0 6px 20px rgba(16,185,129,0.12)";
    document.body.appendChild(el);
    setTimeout(()=> el.remove(), 2600);
  } catch(e){
    alert(msg);
  }
}

async function autoDetectLocation(){
  if(!navigator.geolocation) {
    showToast("Geolocation not available");
    return;
  }
  showBusy(true);
  navigator.geolocation.getCurrentPosition(async (pos)=>{
    const lat = pos.coords.latitude.toFixed(6);
    const lon = pos.coords.longitude.toFixed(6);
    const name = await reverseGeocode(lat, lon);
    if(name) {
      $("locationText").value = name;
      showToast("Location detected: " + name);
      window._detected = { lat: parseFloat(lat), lon: parseFloat(lon) };
    } else {
      $("locationText").value = `${lat}, ${lon}`;
      window._detected = { lat: parseFloat(lat), lon: parseFloat(lon) };
      showToast("Location detected (coords).");
    }
    showBusy(false);
  }, (err)=>{
    showBusy(false);
    showToast("Location denied or failed. Enter location manually.");
  }, {timeout:8000});
}

async function autoFillFeatures(){
  const detected = window._detected || null;
  const lat = detected ? detected.lat : null;
  const lon = detected ? detected.lon : null;
  const payload = { lat, lon, district: $("locationText").value || "" };
  showBusy(true);
  try{
    const res = await fetch(BACKEND + "/auto_features", {
      method:"POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload)
    });
    if(!res.ok) throw new Error("auto_features failed");
    const j = await res.json();
    window._auto = j;
    if(j.source) showToast("Auto-fill done (source: " + j.source + ")");
    else showToast("Auto-fill done");
  }catch(e){
    console.error(e);
    showToast("Auto-fill failed");
  } finally { showBusy(false); }
}

function renderResult(resp){
  $("result").classList.remove("hidden");
  $("yieldBlock").innerHTML = "";
  $("fertBlock").innerHTML = "";
  $("pestBlock").innerHTML = "";
  $("cropBlock").innerHTML = "";

  if(resp.predicted_yield_kg_per_ha !== undefined){
    const areaHa = toHa($("area").value, $("areaUnit").value);
    $("yieldBlock").innerHTML = `<div class="result-row">
      <div class="kv"><strong>Predicted yield (kg/ha)</strong>${resp.predicted_yield_kg_per_ha}</div>
      <div class="kv"><strong>Area (ha)</strong>${areaHa.toFixed(3)} ha</div>
      <div class="kv"><strong>For your area</strong>${Math.round(resp.predicted_yield_kg_per_ha * areaHa)} kg</div>
    </div>`;
  }

  if(resp.fertilizer){
    $("fertBlock").innerHTML = `<div>
      <strong>Fertilizer recommendation</strong>
      <div class="kv"><strong>Product</strong>${escapeHtml(resp.fertilizer.product || resp.fertilizer.company || "—")}</div>
      <div class="kv"><strong>NPK</strong>${resp.fertilizer.npk.join("-")}</div>
      <div class="kv"><strong>Apply (approx)</strong>${resp.delivered.total_kg_per_ha} kg/ha (delivers N=${resp.delivered.N_kg} kg, P=${resp.delivered.P_kg} kg, K=${resp.delivered.K_kg} kg)</div>
    </div>`;
  }

  if(resp.pesticide_advisory){
    $("pestBlock").innerHTML = `<div><strong>Pesticide advisory (top)</strong><pre>${escapeHtml(resp.pesticide_advisory)}</pre></div>`;
  }

  if(resp.recommended_crop || resp.top_candidates){
    const rec = resp.recommended_crop || "";
    const conf = resp.confidence_percent || 0;
    const top = resp.top_candidates || [];
    $("cropBlock").innerHTML = `<div>
      <strong>Crop recommender</strong>
      <div class="kv"><strong>Recommended</strong>${rec} — ${conf}%</div>
      <div class="kv"><strong>Top candidates</strong>${top.map(t=>`${t.crop} (${t.prob}%)`).join(", ")}</div>
    </div>`;
  }
}

function escapeHtml(s){
  return String(s||"").replace(/[&<>"']/g, c=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" })[c]);
}

async function predictYield(){
  const locText = $("locationText").value || "";
  const soilSel = $("soilType").value || null;
  const req = {
    district: locText,
    crop: $("crop").value,
    area_ha: toHa($("area").value, $("areaUnit").value),
    lat: (window._detected && window._detected.lat) || null,
    lon: (window._detected && window._detected.lon) || null,
    soil_type: soilSel
  };
  showBusy(true);
  try{
    const res = await fetch(BACKEND + "/recommend", {
      method:"POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(req)
    });
    if(!res.ok) throw new Error("recommend failed");
    const j = await res.json();
    renderResult(j);
  }catch(e){
    console.error(e);
    showToast("Prediction failed. Check backend.");
  } finally { showBusy(false); }
}

async function predictCrop(){
  const auto = window._auto || {};
  const payload = {
    N: auto.N != null ? auto.N : null,
    P: auto.P != null ? auto.P : null,
    K: auto.K != null ? auto.K : null,
    temperature: auto.temperature != null ? auto.temperature : null,
    humidity: auto.humidity != null ? auto.humidity : null,
    ph: auto.ph != null ? auto.ph : null,
    rainfall: auto.rainfall != null ? auto.rainfall : null
  };
  showBusy(true);
  try{
    const res = await fetch(BACKEND + "/recommend_crop", {
      method:"POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload)
    });
    if(!res.ok) throw new Error("crop predict failed");
    const j = await res.json();
    const out = { recommended_crop: j.recommended_crop, confidence_percent: j.confidence_percent, top_candidates: j.top_candidates };
    renderResult(out);
  }catch(e){
    console.error(e);
    showToast("Crop prediction failed. Check backend.");
  } finally { showBusy(false); }
}

document.addEventListener("DOMContentLoaded", ()=>{
  $("detectBtn").addEventListener("click", (ev)=>{ ev.preventDefault(); autoDetectLocation(); });
  $("autoFillBtn").addEventListener("click", (ev)=>{ ev.preventDefault(); autoFillFeatures(); });
  $("predictYieldBtn").addEventListener("click", (ev)=>{ ev.preventDefault(); predictYield(); });
  $("predictCropBtn").addEventListener("click", (ev)=>{ ev.preventDefault(); predictCrop(); });
});
