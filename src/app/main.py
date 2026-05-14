from fastapi import FastAPI

app = FastAPI(title="Podcast Backend")


@app.get("/")
def root():
    return {"message": "FastAPI is running"}
