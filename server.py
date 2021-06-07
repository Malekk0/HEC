import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
import shutil
from calc import calculated

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/image")
async def result_image(file: UploadFile = File(...)):
    with open(f'{file.filename}', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    path = os.path.join(file.filename)
    result = calculated(path)
    print(result)
    return result


if __name__ == "__main__":
    uvicorn.run("server:app", host="192.168.0.105", port=5000, log_level="info")
