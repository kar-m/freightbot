import json

class freight_company:
    air: bool
    weight_: float
    value_: float
    boxes_: float
    pallets_: float
    distance_: float
    name: str

    def __init__(self, a:bool, w:float, v:float, b:float, p:float, d:float, name_:str):
        self.air = a 
        self.weight_ = w 
        self.value_ = v 
        self.boxes_ = b 
        self.pallets_ = p 
        self.distance_ = d 
        self.name = name_

    def get_quote(self, w:float, v:float, b:float, p:float, d:float) -> float:
        weight_p=1
        distance_p=1
        if self.air:
            weight_p = 2
            distance_p=0.5

        q = self.weight_* (w**weight_p) + self.value_*v + self.boxes_*b+self.pallets_*p + self.distance_*(d**distance_p) / 100
        q /= 1000

        q = int(q)

        return q
    

class Engine():
    companies: list[freight_company]

    def __init__(self):
        self.companies = [
            freight_company(True, 2.5, 3.8, 1.4, 4.9, 2.7, "Flying Crates Inc."),
            freight_company(False, 4.1, 2.3, 3.7, 1.2, 4.6, "Whimsical Wheels"),
            freight_company(True, 1.9, 4.5, 2.3, 3.3, 1.8, "Cargo Craze"),
            freight_company(False, 3.3, 1.6, 4.2, 2.7, 3.5, "Freight Fiesta"),
            freight_company(True, 2.7, 2.9, 3.1, 1.8, 4.1, "Boxy Business"),
            freight_company(False, 4.4, 1.2, 2.8, 3.9, 1.5, "Pallet Parade"),
            freight_company(True, 1.3, 4.2, 3.5, 2.1, 4.8, "Airy Adventures"),
            freight_company(False, 3.6, 3.1, 1.9, 4.4, 2.2, "Haul Hilarity"),
            freight_company(True, 2.2, 2.4, 4.7, 1.5, 3.0, "Quirky Cargo Co."),
            freight_company(False, 3.0, 1.8, 2.6, 3.4, 4.3, "Pallet Pals")
        ]

    
    def read_json(self, json_input:str):
        try:
            json_dict = json.loads(json_input)
        except:
            print("json_generation_failed")
            return -1
        return [
            json_dict['air'],
            json_dict['weight'],
            json_dict['value'],
            json_dict['boxes'],
            json_dict['pallets'],
            json_dict['distance']
        ]

    def get_quotes(self, json_input:str) -> str:
        inputs = self.read_json(json_input)
        if inputs==-1:
            return -1
        filtered_companies = []
        for cmp in self.companies:
            if inputs[0] == cmp.air:
                filtered_companies.append(cmp)
        quote_outputs = [{'company_name':cmp.name, 'quote':cmp.get_quote(*inputs[1:])} for cmp in filtered_companies]
        return json.dumps(quote_outputs)
