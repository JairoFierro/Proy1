from pydantic import BaseModel

class DataModel(BaseModel):
    year: float
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str

    def columns(self):
        return ["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
