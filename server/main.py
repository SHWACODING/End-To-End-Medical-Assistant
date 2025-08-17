from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_chack():
    return {"status": "ok"}



def main():
    print("Hello from server!")


if __name__ == "__main__":
    main()
