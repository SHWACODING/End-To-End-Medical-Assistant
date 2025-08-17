from fastapi import FastAPI
from auth.routes import router as auth_router
from docs.routes import router as docs_router



app = FastAPI()


app.include_router(auth_router)
app.include_router(docs_router)


@app.get("/health")
def health_chack():
    return {"status": "ok"}

