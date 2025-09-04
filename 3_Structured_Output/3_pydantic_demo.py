from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = 'nitish'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')


new_student = {'age':'32', 'email':'abc@gmail.com'}

# pydentic object
student = Student(**new_student)

# converting pydantic object into python dictionary with 2 methods
# student_dict = dict(student)
# or
student_dict =student.model_dump()

# converting pydantic object into python dictionary
student_json =student.model_dump_json()

print(student)
print(student.name) # object

print(student_dict["age"]) # dictionary

print(student_json)
print(student_json)