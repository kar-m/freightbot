import reflex as rx

class DataTable(rx.Base):
    cols: list[dict] = [
        {
            "title": "Name",
            "type": "str",
            "width": 150,
        },
        {
            "title": "Weight",
            "type": "float",
            "width": 80,
        },
        {
            "title": "Length",
            "type": "float",
            "width": 80,
        },
        {
            "title": "Width",
            "type": "float",
            "width": 80,
        },
        {
            "title": "Depth",
            "type": "float",
            "width": 80,
        },
        {
            "title": "Quantity",
            "type": "int",
            "width": 80,
        },
    ]
    data: list
    change_logs: list

    def apply_update(self, update_list: list):
        self.change_logs.append(update_list)
        for i, row in enumerate(self.data):
            if row[0] == update_list[0]:
                for j in range(len(row)):
                    if not (update_list[j] is None):
                        self.data[i][j] = update_list[j]

                return
        
        self.data.append(update_list)

    def __str__(self):
        return str(self.data)

    def get_edited_data(self, pos, val):
        col, row = pos
        self.data[row][col] = val["data"]
        update_list = [self.data[row][0], None,None,None,None,None]
        update_list[col] = val["data"]
        self.change_logs = update_list

