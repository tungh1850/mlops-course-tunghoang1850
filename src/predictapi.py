import uvicorn
from fastapi import FastAPI

from src.router.predict import router

app = FastAPI()
app.include_router(router)


# set root endpoint
@app.get("/")
def root() -> dict:
    return {"message": "Credit Risk Prediction API is running."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
