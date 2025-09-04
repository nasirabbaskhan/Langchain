from typing import TypedDict

class Person(TypedDict):
    name: str
    age:int 


new_person:Person = {"name":"nasir", "age":"24"}

print(new_person) # it does not validate the type 