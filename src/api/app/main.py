from fastapi import FastAPI
from api.routers import train, predict, evaluate

app = FastAPI()

app.include_router(train.router)
app.include_router(predict.router)
app.include_router(evaluate.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)