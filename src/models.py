from pydantic import BaseModel, Field
from typing import List

class Exercise(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    sets: int = Field(ge=1, le=20)
    reps: int = Field(ge=1, le=100)

class WorkoutDay(BaseModel):
    day: str = Field(min_length=1, max_length=50)
    exercises: List[Exercise] = Field(min_length=1)

class DietPlan(BaseModel):
    calories: int = Field(ge=500, le=10000)
    protein: int = Field(ge=10, le=500)

class ApprovedSchema(BaseModel):
    workout_days: List[WorkoutDay] = Field(min_length=1)
    diet: DietPlan
