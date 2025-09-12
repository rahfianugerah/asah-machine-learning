import os, time, json, threading, requests

from prometheus_client import (
    start_http_server, CollectorRegistry, Counter, Gauge, Histogram, Summary
)

API_URL      = os.getenv("API_URL", "http://inference:8000/predict")
HTTP_METHOD  = os.getenv("HTTP_METHOD", "POST").upper()
TIMEOUT      = float(os.getenv("TIMEOUT_SEC", "5"))
INTERVAL     = float(os.getenv("INTERVAL_SEC", "2"))
EXPORTER_PORT= int(os.getenv("EXPORTER_PORT", "8001"))
DEFAULT_PAYLOAD = os.getenv("PAYLOAD_JSON", json.dumps({"X":[[1,2,3,4,5]]}))

REG = CollectorRegistry()
REQ_TOTAL   = Counter("inference_requests_total", "Total hits", registry=REG)
REQ_SUCCESS = Counter("inference_success_total", "Success", registry=REG)
REQ_ERROR   = Counter("inference_error_total", "Errors", registry=REG)
INPROG      = Gauge("inference_in_progress", "In-flight", registry=REG)
LATENCY     = Histogram("inference_latency_seconds", "Latency (s)",
                        buckets=(0.01,0.05,0.1,0.2,0.5,1,2,5), registry=REG)
BATCH_HIST  = Histogram("inference_input_batch", "Batch size",
                        buckets=(1,2,4,8,16,32), registry=REG)
PAYLOAD_SUM = Summary("inference_payload_bytes", "Payload bytes", registry=REG)

ACC=Gauge("model_accuracy","acc",registry=REG)
PREC=Gauge("model_precision","prec",registry=REG)
REC=Gauge("model_recall","recall",registry=REG)
F1=Gauge("model_f1","f1",registry=REG)
TP=Gauge("model_confusion_tp","TP",registry=REG)
TN=Gauge("model_confusion_tn","TN",registry=REG)
FP=Gauge("model_confusion_fp","FP",registry=REG)
FN=Gauge("model_confusion_fn","FN",registry=REG)

def seed_quality():
    for k,g in [("ACC",ACC),("PREC",PREC),("RECALL",REC),("F1",F1),
                ("TP",TP),("TN",TN),("FP",FP),("FN",FN)]:
        v=os.getenv(k)
        if v is not None:
            try: g.set(float(v))
            except: pass

def guess_batch(payload_str: str) -> int:
    try:
        obj=json.loads(payload_str); X=obj.get("X",[])
        return len(X) if isinstance(X,list) else 1
    except: return 1

def hit_once(payload: str):
    REQ_TOTAL.inc(); INPROG.inc()
    t0=time.time()
    try:
        if HTTP_METHOD=="GET":
            r=requests.get(API_URL, timeout=TIMEOUT)
        else:
            PAYLOAD_SUM.observe(len(payload.encode()))
            BATCH_HIST.observe(guess_batch(payload))
            r=requests.post(API_URL, data=payload, headers={"Content-Type":"application/json"}, timeout=TIMEOUT)
        LATENCY.observe(time.time()-t0)
        if 200 <= r.status_code < 300:
            REQ_SUCCESS.inc(); print(f"[INFO] OK {r.status_code}")
        else:
            REQ_ERROR.inc();   print(f"[WARN] HTTP {r.status_code}")
    except Exception as e:
        LATENCY.observe(time.time()-t0); REQ_ERROR.inc()
        print(f"[ERROR] {e}")
    finally:
        INPROG.dec()

def loop():
    payload=DEFAULT_PAYLOAD
    while True:
        hit_once(payload)
        time.sleep(max(0.0, INTERVAL))

if __name__=="__main__":
    seed_quality()
    start_http_server(EXPORTER_PORT, registry=REG)
    threading.Thread(target=loop, daemon=True).start()
    while True: time.sleep(3600)