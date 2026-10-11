import os
import json
import base64

from io import BytesIO
from typing import TypedDict, List

import streamlit as st

from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import TA_CENTER

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:

    st.error(
        "OPENAI_API_KEY is missing in your .env file."
    )

    st.stop()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="DietAI",

    page_icon="🥗",

    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .scanner-title {
        font-size: 30px;
        font-weight: 700;
    }

    .nutrition-card {
        padding: 18px;
        border-radius: 15px;
        background-color: #f7f7f7;
        text-align: center;
    }

    .small-text {
        color: #6b7280;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD FOOD DATABASE
# ============================================================

@st.cache_data
def load_foods():

    try:

        with open(
            "foods.json",
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except FileNotFoundError:

        st.error(
            "foods.json was not found. "
            "Place foods.json in the same folder as app.py."
        )

        st.stop()


FOODS = load_foods()


FOOD_LOOKUP = {

    food["id"]: food

    for food in FOODS
}


# ============================================================
# PYDANTIC MODELS
# ============================================================


class DayPlan(BaseModel):

    day: str

    breakfast: str

    mid_morning: str

    lunch: str

    evening_snack: str

    dinner: str

    estimated_calories: int

    estimated_protein: int


class DietPlan(BaseModel):

    summary: str

    recommendations: List[str]

    weekly_plan: List[DayPlan]


class ReplacementResult(BaseModel):

    replacement_food_id: str

    replacement_meal: str

    reason: str


class GroceryList(BaseModel):

    grains: List[str]

    proteins: List[str]

    vegetables: List[str]

    fruits: List[str]

    dairy: List[str]

    other: List[str]


# ============================================================
# FOOD SCANNER MODEL
# ============================================================


class FoodAnalysis(BaseModel):

    food_name: str = Field(
        description="Name of the detected food or meal."
    )

    description: str = Field(
        description="Short description of what is visible."
    )

    estimated_weight_g: float

    calories: float

    protein_g: float

    carbohydrates_g: float

    fat_g: float

    fiber_g: float

    sugar_g: float

    saturated_fat_g: float

    sodium_mg: float

    potassium_mg: float

    calcium_mg: float

    iron_mg: float

    magnesium_mg: float

    zinc_mg: float

    phosphorus_mg: float

    vitamin_a_mcg: float

    vitamin_c_mg: float

    vitamin_d_mcg: float

    vitamin_e_mg: float

    vitamin_k_mcg: float

    vitamin_b1_mg: float

    vitamin_b2_mg: float

    vitamin_b3_mg: float

    vitamin_b6_mg: float

    vitamin_b12_mcg: float

    folate_mcg: float

    confidence: str

    assumptions: List[str]


# ============================================================
# KITCHEN SCANNER MODELS
# ============================================================

class KitchenIngredient(BaseModel):
    name: str = Field(description="Food ingredient visible in the kitchen.")
    category: str = Field(description="Category such as grain, vegetable, fruit, protein, dairy, spice, etc.")
    confidence: str = Field(description="High, Medium or Low confidence.")


class KitchenScanResult(BaseModel):
    ingredients: List[KitchenIngredient]


class KitchenRecipe(BaseModel):
    name: str
    description: str
    cooking_time_minutes: int
    servings: int
    calories_per_serving: int
    protein_g: float
    carbohydrates_g: float
    fat_g: float
    fiber_g: float
    ingredients: List[str]
    instructions: List[str]
    missing_ingredients: List[str]


class KitchenRecipeResult(BaseModel):
    recipes: List[KitchenRecipe]


# ============================================================
# LANGGRAPH STATE
# ============================================================


class DietState(TypedDict):

    name: str

    age: int

    sex: str

    height_cm: float

    weight_kg: float

    bmi: float

    bmi_category: str

    bmr: int

    tdee: int

    calories: int

    protein_target: int

    goal: str

    diet_type: str

    cuisine: str

    activity: str

    food_candidates: str

    validation_attempt: int

    validation_message: str

    diet_plan: dict


# ============================================================
# OPENAI MODELS
# ============================================================


llm = ChatOpenAI(

    model="gpt-5.5",

    temperature=0.3,

    api_key=OPENAI_API_KEY
)


structured_diet_llm = (
    llm.with_structured_output(
        DietPlan
    )
)


structured_replacement_llm = (
    llm.with_structured_output(
        ReplacementResult
    )
)


structured_grocery_llm = (
    llm.with_structured_output(
        GroceryList
    )
)


# Vision model

food_vision_llm = ChatOpenAI(

    model="gpt-5.5",

    temperature=0,

    api_key=OPENAI_API_KEY
)


structured_food_vision_llm = (
    food_vision_llm.with_structured_output(
        FoodAnalysis
    )
)


structured_kitchen_llm = (
    food_vision_llm.with_structured_output(
        KitchenScanResult
    )
)

structured_kitchen_recipe_llm = (
    llm.with_structured_output(
        KitchenRecipeResult
    )
)


# ============================================================
# BMI
# ============================================================


def calculate_bmi(
    state: DietState
):

    height_m = (
        state["height_cm"]
        /
        100
    )

    bmi = (
        state["weight_kg"]
        /
        (height_m ** 2)
    )

    bmi = round(
        bmi,
        2
    )


    if bmi < 18.5:

        category = "Underweight"

    elif bmi < 25:

        category = "Normal range"

    elif bmi < 30:

        category = "Overweight range"

    else:

        category = "Obesity range"


    return {

        "bmi": bmi,

        "bmi_category": category
    }


# ============================================================
# BMR + TDEE
# ============================================================


def calculate_energy(
    state: DietState
):

    weight = state["weight_kg"]

    height = state["height_cm"]

    age = state["age"]

    sex = state["sex"]


    if sex == "Male":

        bmr = (

            10 * weight

            +

            6.25 * height

            -

            5 * age

            +

            5
        )

    elif sex == "Female":

        bmr = (

            10 * weight

            +

            6.25 * height

            -

            5 * age

            -

            161
        )

    else:

        bmr = (

            10 * weight

            +

            6.25 * height

            -

            5 * age

            -

            78
        )


    activity_factors = {

        "Sedentary": 1.20,

        "Lightly Active": 1.375,

        "Moderately Active": 1.55,

        "Very Active": 1.725
    }


    tdee = (

        bmr

        *

        activity_factors[
            state["activity"]
        ]
    )


    if state["goal"] == "Get Lean":

        target = tdee - 300

    elif state["goal"] == "Bulk":

        target = tdee + 300

    else:

        target = tdee


    target = max(
        target,
        1400
    )


    if state["goal"] == "Bulk":

        protein = (
            state["weight_kg"]
            * 1.6
        )

    elif state["goal"] == "Get Lean":

        protein = (
            state["weight_kg"]
            * 1.5
        )

    else:

        protein = (
            state["weight_kg"]
            * 1.2
        )


    return {

        "bmr": int(bmr),

        "tdee": int(tdee),

        "calories": int(target),

        "protein_target": int(
            protein
        )
    }


# ============================================================
# FILTER FOODS
# ============================================================


def filter_foods(
    state: DietState
):

    diet = state["diet_type"]

    cuisine = state["cuisine"]

    candidates = []


    for food in FOODS:

        if diet == "Vegetarian":

            if food["diet"] != "Vegetarian":

                continue


        elif diet == "Eggetarian":

            if food["diet"] not in [

                "Vegetarian",

                "Eggetarian"
            ]:

                continue


        if cuisine != "Any":

            if (

                food["cuisine"]
                != cuisine

                and

                food["cuisine"]
                != "Indian"
            ):

                continue


        candidates.append(
            food
        )


    candidates = candidates[:60]


    food_text = "\n".join(

        [

            (
                f'ID={food["id"]} | '
                f'Name={food["name"]} | '
                f'Meal={food["meal"]} | '
                f'Serving={food["serving"]} | '
                f'Calories={food["calories"]} | '
                f'Protein={food["protein"]}g | '
                f'Carbs={food["carbs"]}g | '
                f'Fat={food["fat"]}g'
            )

            for food in candidates
        ]
    )


    return {

        "food_candidates":
            food_text
    }


# ============================================================
# GENERATE DIET PLAN
# ============================================================


def generate_plan(
    state: DietState
):

    prompt = f"""

You are DietAI.

Create a practical 7-day Indian meal plan.

USER
====

Name:
{state["name"]}

Age:
{state["age"]}

Sex:
{state["sex"]}

Height:
{state["height_cm"]} cm

Weight:
{state["weight_kg"]} kg

BMI:
{state["bmi"]}

BMR:
{state["bmr"]}

TDEE:
{state["tdee"]}

Daily calorie target:
{state["calories"]}

Daily protein target:
{state["protein_target"]} g

Goal:
{state["goal"]}

Diet:
{state["diet_type"]}

Cuisine:
{state["cuisine"]}

Activity:
{state["activity"]}


FOOD DATABASE
=============

{state["food_candidates"]}


RULES
=====

1. Create exactly 7 days.

2. Every day must contain:

Breakfast
Mid-morning snack
Lunch
Evening snack
Dinner

3. Prefer foods from the database.

4. Respect diet type.

5. Respect cuisine.

6. Keep calories reasonably close
to the daily target.

7. Keep protein reasonably close
to the daily target.

8. For Get Lean:
prioritize protein, vegetables
and reasonable portions.

9. For Bulk:
use nutritious calorie-dense foods.

10. For Maintain:
create a balanced diet.

11. Avoid extreme diets.

12. Do not provide medical diagnosis.

13. Do not make medical treatment claims.

14. estimated_calories and
estimated_protein should be reasonable
estimates.

15. Return exactly seven days.
"""


    result = structured_diet_llm.invoke(
        prompt
    )


    return {

        "diet_plan":
            result.model_dump()
    }


# ============================================================
# VALIDATE PLAN
# ============================================================


def validate_plan(
    state: DietState
):

    plan = state[
        "diet_plan"
    ]

    days = plan[
        "weekly_plan"
    ]


    avg_calories = (

        sum(

            day[
                "estimated_calories"
            ]

            for day in days
        )

        /

        len(days)
    )


    avg_protein = (

        sum(

            day[
                "estimated_protein"
            ]

            for day in days
        )

        /

        len(days)
    )


    calorie_difference = abs(

        avg_calories
        -
        state["calories"]
    )


    protein_difference = abs(

        avg_protein
        -
        state["protein_target"]
    )


    calories_valid = (

        calorie_difference
        <=
        state["calories"] * 0.15
    )


    protein_valid = (

        protein_difference
        <=
        max(
            state["protein_target"]
            * 0.25,

            15
        )
    )


    if (

        calories_valid
        and
        protein_valid
    ):

        message = (

            "Plan validated. "

            f"Average calories: "
            f"{avg_calories:.0f}. "

            f"Average protein: "
            f"{avg_protein:.0f}g."
        )

    else:

        message = (

            "Plan needs improvement. "

            f"Target calories: "
            f"{state['calories']}. "

            f"Generated: "
            f"{avg_calories:.0f}. "

            f"Target protein: "
            f"{state['protein_target']}g. "

            f"Generated: "
            f"{avg_protein:.0f}g."
        )


    return {

        "validation_message":
            message
    }


# ============================================================
# VALIDATION ROUTER
# ============================================================


def validation_router(
    state: DietState
):

    if (

        "needs improvement"
        in
        state["validation_message"]

        and

        state["validation_attempt"]
        < 1
    ):

        return "regenerate"


    return "finish"


# ============================================================
# REGENERATE PLAN
# ============================================================


def regenerate_plan(
    state: DietState
):

    prompt = f"""

Improve this meal plan.

Target calories:
{state["calories"]}

Target protein:
{state["protein_target"]}g

Goal:
{state["goal"]}

Diet:
{state["diet_type"]}

Cuisine:
{state["cuisine"]}

Validation:
{state["validation_message"]}

Previous plan:

{json.dumps(
    state["diet_plan"],
    indent=2
)}

Improve calorie and protein alignment.

Keep exactly 7 days.

Return the complete plan.
"""


    result = structured_diet_llm.invoke(
        prompt
    )


    return {

        "diet_plan":
            result.model_dump(),

        "validation_attempt":
            state["validation_attempt"]
            + 1
    }


# ============================================================
# LANGGRAPH
# ============================================================


def build_graph():

    graph = StateGraph(
        DietState
    )


    graph.add_node(
        "calculate_bmi",
        calculate_bmi
    )

    graph.add_node(
        "calculate_energy",
        calculate_energy
    )

    graph.add_node(
        "filter_foods",
        filter_foods
    )

    graph.add_node(
        "generate_plan",
        generate_plan
    )

    graph.add_node(
        "validate_plan",
        validate_plan
    )

    graph.add_node(
        "regenerate_plan",
        regenerate_plan
    )


    graph.add_edge(
        START,
        "calculate_bmi"
    )

    graph.add_edge(
        "calculate_bmi",
        "calculate_energy"
    )

    graph.add_edge(
        "calculate_energy",
        "filter_foods"
    )

    graph.add_edge(
        "filter_foods",
        "generate_plan"
    )

    graph.add_edge(
        "generate_plan",
        "validate_plan"
    )


    graph.add_conditional_edges(

        "validate_plan",

        validation_router,

        {

            "regenerate":
                "regenerate_plan",

            "finish":
                END
        }
    )


    graph.add_edge(
        "regenerate_plan",
        "validate_plan"
    )


    return graph.compile()


diet_graph = build_graph()


# ============================================================
# NUTRITION ENGINE
# ============================================================


def calculate_meal_nutrition(
    meal_text
):

    if not meal_text:

        return {

            "calories": 0,

            "protein": 0,

            "carbs": 0,

            "fat": 0,

            "foods": []
        }


    text = (
        meal_text
        .lower()
    )


    calories = 0

    protein = 0

    carbs = 0

    fat = 0

    matched_foods = []


    for food in FOODS:

        food_name = (
            food["name"]
            .lower()
        )


        if food_name in text:

            calories += (
                food["calories"]
            )

            protein += (
                food["protein"]
            )

            carbs += (
                food["carbs"]
            )

            fat += (
                food["fat"]
            )

            matched_foods.append(
                food["name"]
            )


    return {

        "calories": calories,

        "protein": protein,

        "carbs": carbs,

        "fat": fat,

        "foods": matched_foods
    }


# ============================================================
# DAY NUTRITION
# ============================================================


def calculate_day_nutrition(
    day
):

    meal_names = [

        "breakfast",

        "mid_morning",

        "lunch",

        "evening_snack",

        "dinner"
    ]


    total = {

        "calories": 0,

        "protein": 0,

        "carbs": 0,

        "fat": 0
    }


    meal_nutrition = {}


    for meal in meal_names:

        nutrition = (
            calculate_meal_nutrition(
                day.get(
                    meal,
                    ""
                )
            )
        )


        meal_nutrition[
            meal
        ] = nutrition


        total["calories"] += (
            nutrition["calories"]
        )

        total["protein"] += (
            nutrition["protein"]
        )

        total["carbs"] += (
            nutrition["carbs"]
        )

        total["fat"] += (
            nutrition["fat"]
        )


    return {

        "total": total,

        "meals":
            meal_nutrition
    }


# ============================================================
# WEEK NUTRITION
# ============================================================


def calculate_week_nutrition(
    plan
):

    weekly = []


    total = {

        "calories": 0,

        "protein": 0,

        "carbs": 0,

        "fat": 0
    }


    for day in plan[
        "weekly_plan"
    ]:

        nutrition = (
            calculate_day_nutrition(
                day
            )
        )


        weekly.append({

            "day":
                day["day"],

            **nutrition[
                "total"
            ]
        })


        total["calories"] += (
            nutrition[
                "total"
            ]["calories"]
        )

        total["protein"] += (
            nutrition[
                "total"
            ]["protein"]
        )

        total["carbs"] += (
            nutrition[
                "total"
            ]["carbs"]
        )

        total["fat"] += (
            nutrition[
                "total"
            ]["fat"]
        )


    return {

        "days": weekly,

        "total": total
    }


# ============================================================
# NUTRITION SCORE
# ============================================================


def nutrition_score(

    daily,

    calorie_target,

    protein_target,

    day=None
):

    score = 0


    # --------------------------------------------------------
    # CALORIES - 30
    # --------------------------------------------------------

    calories = daily.get(
        "calories",
        0
    )


    if calorie_target > 0:

        ratio = (
            calories
            /
            calorie_target
        )


        if 0.95 <= ratio <= 1.05:

            calorie_score = 30

        elif 0.90 <= ratio <= 1.10:

            calorie_score = 27

        elif 0.85 <= ratio <= 1.15:

            calorie_score = 23

        elif 0.75 <= ratio <= 1.25:

            calorie_score = 18

        else:

            calorie_score = 10

    else:

        calorie_score = 0


    score += calorie_score


    # --------------------------------------------------------
    # PROTEIN - 30
    # --------------------------------------------------------

    protein = daily.get(
        "protein",
        0
    )


    if protein_target > 0:

        ratio = (
            protein
            /
            protein_target
        )


        if ratio >= 1.0:

            protein_score = 30

        elif ratio >= 0.90:

            protein_score = 27

        elif ratio >= 0.80:

            protein_score = 23

        elif ratio >= 0.70:

            protein_score = 18

        else:

            protein_score = 10

    else:

        protein_score = 0


    score += protein_score


    # --------------------------------------------------------
    # MACRO BALANCE - 15
    # --------------------------------------------------------

    carbs = daily.get(
        "carbs",
        0
    )

    fat = daily.get(
        "fat",
        0
    )


    macro_calories = (

        protein * 4

        +

        carbs * 4

        +

        fat * 9
    )


    macro_score = 0


    if macro_calories > 0:

        protein_pct = (
            protein * 4
        ) / macro_calories


        carb_pct = (
            carbs * 4
        ) / macro_calories


        fat_pct = (
            fat * 9
        ) / macro_calories


        if 0.15 <= protein_pct <= 0.35:

            macro_score += 5

        elif 0.10 <= protein_pct <= 0.40:

            macro_score += 3


        if 0.35 <= carb_pct <= 0.60:

            macro_score += 5

        elif 0.25 <= carb_pct <= 0.65:

            macro_score += 3


        if 0.15 <= fat_pct <= 0.35:

            macro_score += 5

        elif 0.10 <= fat_pct <= 0.40:

            macro_score += 3


    score += macro_score


    # --------------------------------------------------------
    # FOOD VARIETY - 15
    # --------------------------------------------------------

    variety_score = 0


    if day:

        meal_names = [

            "breakfast",

            "mid_morning",

            "lunch",

            "evening_snack",

            "dinner"
        ]


        text = " ".join(

            str(
                day.get(
                    meal,
                    ""
                )
            )

            for meal in meal_names
        ).lower()


        matched = set()


        for food in FOODS:

            if (
                food["name"]
                .lower()
                in text
            ):

                matched.add(
                    food["id"]
                )


        count = len(
            matched
        )


        if count >= 10:

            variety_score = 15

        elif count >= 8:

            variety_score = 13

        elif count >= 6:

            variety_score = 10

        elif count >= 4:

            variety_score = 7

        else:

            variety_score = 4


    score += variety_score


    # --------------------------------------------------------
    # MEAL COVERAGE - 10
    # --------------------------------------------------------

    meal_score = 0


    if day:

        required = [

            "breakfast",

            "mid_morning",

            "lunch",

            "evening_snack",

            "dinner"
        ]


        completed = 0


        for meal in required:

            if str(
                day.get(
                    meal,
                    ""
                )
            ).strip():

                completed += 1


        meal_score = (

            completed
            /
            len(required)
            *
            10
        )


    score += meal_score


    return max(

        0,

        min(
            100,
            round(score)
        )
    )


# ============================================================
# FOOD SCANNER
# ============================================================


def analyze_food_image(

    image_bytes,

    image_type,

    weight_g
):

    image_base64 = (
        base64.b64encode(
            image_bytes
        )
        .decode(
            "utf-8"
        )
    )


    prompt = f"""

You are DietAI Food Scanner.

Analyze the uploaded food photograph.

The user has provided an approximate
TOTAL FOOD WEIGHT of:

{weight_g} grams.


YOUR TASK
=========

Identify the food or foods visible.

Then estimate the nutrition for the
provided total weight.


Return:

1. Food name

2. Short description

3. Estimated weight

4. Calories

5. Macronutrients:

Protein
Carbohydrates
Fat
Fiber
Sugar
Saturated Fat

6. Minerals:

Sodium
Potassium
Calcium
Iron
Magnesium
Zinc
Phosphorus

7. Vitamins:

Vitamin A
Vitamin C
Vitamin D
Vitamin E
Vitamin K

Vitamin B1
Vitamin B2
Vitamin B3
Vitamin B6
Vitamin B12

Folate


IMPORTANT
=========

This is an IMAGE-BASED ESTIMATE.

A photograph cannot reliably determine:

- exact grams of each ingredient
- exact cooking oil quantity
- exact recipe
- exact calories
- exact micronutrients

Therefore:

- Give reasonable estimates.
- Do not pretend they are laboratory measurements.
- State assumptions.
- Give a confidence level.
- If the food cannot be identified reliably,
say so.
- If multiple foods are visible,
analyze the complete meal.
- Consider the user-provided weight.
- Consider common Indian preparation methods.
- Account for visible oil, gravy or sauces
when reasonably possible.

Do not provide medical diagnosis.

This is for general wellness and
educational purposes only.
"""


    message = [

        {
            "type": "text",

            "text": prompt
        },

        {
            "type": "image_url",

            "image_url": {

                "url":
                (
                    f"data:{image_type};base64,"
                    f"{image_base64}"
                )
            }
        }
    ]


    response = (
        structured_food_vision_llm.invoke(
            [
                {
                    "role": "user",

                    "content": message
                }
            ]
        )
    )


    return response.model_dump()


# ============================================================
# KITCHEN SCANNER
# ============================================================

def scan_kitchen_images(uploaded_files):
    """
    Accept multiple kitchen/refrigerator/pantry photos and use
    the OpenAI vision model to identify visible food ingredients.
    """
    content = [{
        "type": "text",
        "text": """
You are DietAI Kitchen Scanner.

Analyze all uploaded kitchen photos and identify FOOD INGREDIENTS
that are visibly present.

Look for:
- rice, atta, rava, oats and other grains
- dal, pulses and legumes
- vegetables
- fruits
- eggs
- paneer, milk, curd and other dairy
- meat and fish
- nuts and seeds
- spices and common cooking ingredients
- bread and packaged food when clearly visible

Rules:
1. Identify only items you can reasonably see.
2. Do not invent ingredients.
3. Avoid duplicates across photos.
4. Use simple ingredient names.
5. Give confidence as High, Medium or Low.
6. Do not identify non-food objects as ingredients.
"""
    }]

    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = uploaded_file.type or "image/jpeg"

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_b64}"
            }
        })

    response = structured_kitchen_llm.invoke(
        [{"role": "user", "content": content}]
    )

    return response.model_dump()


def generate_kitchen_recipes(
    ingredients,
    result,
    cooking_time,
    preference
):
    ingredient_text = ", ".join(ingredients)

    prompt = f"""
You are DietAI Recipe Chef.

The user has scanned their kitchen and confirmed these ingredients:

{ingredient_text}

USER PROFILE
============
Goal: {result["goal"]}
Diet: {result["diet_type"]}
Cuisine: {result["cuisine"]}
Daily calorie target: {result["calories"]} kcal
Daily protein target: {result["protein_target"]} g

Maximum cooking time:
{cooking_time} minutes

User preference:
{preference or "No special preference"}

Create EXACTLY 3 practical recipes.

RULES
=====
1. Use the available ingredients as the main ingredients.
2. Respect the user's diet type.
3. Respect the selected cuisine.
4. If South Indian is selected, prefer South Indian recipes.
5. For Get Lean, prioritize protein, vegetables and fiber.
6. For Bulk, make the recipe more calorie/protein dense.
7. For Maintain, keep it balanced.
8. Keep cooking time within the requested limit where practical.
9. Clearly list missing ingredients separately.
10. Never claim a missing ingredient is available.
11. Common basics such as salt, spices and cooking oil may be assumed,
    but still list them if they are important.
12. Give approximate calories and macros per serving.
13. Give clear step-by-step cooking instructions.
14. Do not provide medical diagnosis or treatment.
"""

    response = structured_kitchen_recipe_llm.invoke(prompt)
    return response.model_dump()


# ============================================================
# PDF
# ============================================================


def create_pdf(

    result,

    nutrition
):

    buffer = BytesIO()


    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        leftMargin=35,

        rightMargin=35,

        topMargin=35,

        bottomMargin=35
    )


    styles = (
        getSampleStyleSheet()
    )


    title_style = ParagraphStyle(

        "TitleStyle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=23,

        spaceAfter=20
    )


    heading_style = ParagraphStyle(

        "HeadingStyle",

        parent=styles["Heading2"],

        fontSize=16,

        spaceBefore=15,

        spaceAfter=8
    )


    day_style = ParagraphStyle(

        "DayStyle",

        parent=styles["Heading3"],

        fontSize=13,

        spaceBefore=12,

        spaceAfter=7
    )


    body_style = ParagraphStyle(

        "BodyStyle",

        parent=styles["BodyText"],

        fontSize=9,

        leading=13
    )


    story = []


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(

        Paragraph(
            "DietAI",
            title_style
        )
    )


    story.append(

        Paragraph(

            f"Personalized Meal Plan for "
            f"{result['name']}",

            body_style
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    story.append(

        Paragraph(
            "Profile & Targets",
            heading_style
        )
    )


    profile = [

        [
            "Age",
            str(result["age"])
        ],

        [
            "Height",
            f'{result["height_cm"]:.1f} cm'
        ],

        [
            "Weight",
            f'{result["weight_kg"]} kg'
        ],

        [
            "BMI",
            str(result["bmi"])
        ],

        [
            "BMI Category",
            result["bmi_category"]
        ],

        [
            "BMR",
            f'{result["bmr"]} kcal'
        ],

        [
            "TDEE",
            f'{result["tdee"]} kcal'
        ],

        [
            "Daily Calories",
            f'{result["calories"]} kcal'
        ],

        [
            "Protein Target",
            f'{result["protein_target"]} g'
        ],

        [
            "Goal",
            result["goal"]
        ],

        [
            "Diet",
            result["diet_type"]
        ],

        [
            "Cuisine",
            result["cuisine"]
        ]
    ]


    profile_table = Table(

        profile,

        colWidths=[
            150,
            300
        ]
    )


    profile_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )


    story.append(
        profile_table
    )


    plan = result[
        "diet_plan"
    ]


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    story.append(

        Paragraph(
            "Plan Summary",
            heading_style
        )
    )


    story.append(

        Paragraph(
            plan["summary"],
            body_style
        )
    )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    story.append(

        Paragraph(
            "Recommendations",
            heading_style
        )
    )


    for recommendation in plan[
        "recommendations"
    ]:

        story.append(

            Paragraph(
                f"- {recommendation}",
                body_style
            )
        )


    # --------------------------------------------------------
    # NUTRITION SUMMARY
    # --------------------------------------------------------

    story.append(

        Paragraph(
            "Weekly Nutrition",
            heading_style
        )
    )


    nutrition_rows = [

        [
            "Day",
            "Calories",
            "Protein",
            "Carbs",
            "Fat"
        ]
    ]


    for day in nutrition[
        "days"
    ]:

        nutrition_rows.append([

            day["day"],

            f'{day["calories"]} kcal',

            f'{day["protein"]} g',

            f'{day["carbs"]} g',

            f'{day["fat"]} g'
        ])


    nutrition_table = Table(

        nutrition_rows,

        colWidths=[
            90,
            100,
            90,
            90,
            70
        ]
    )


    nutrition_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )


    story.append(
        nutrition_table
    )


    # --------------------------------------------------------
    # 7 DAY PLAN
    # --------------------------------------------------------

    story.append(

        Paragraph(
            "7-Day Meal Plan",
            heading_style
        )
    )


    for day in plan[
        "weekly_plan"
    ]:

        story.append(

            Paragraph(
                day["day"],
                day_style
            )
        )


        meal_data = [

            [
                "Meal",
                "Food"
            ],

            [
                "Breakfast",
                day["breakfast"]
            ],

            [
                "Mid Morning",
                day["mid_morning"]
            ],

            [
                "Lunch",
                day["lunch"]
            ],

            [
                "Evening Snack",
                day["evening_snack"]
            ],

            [
                "Dinner",
                day["dinner"]
            ]
        ]


        meal_table = Table(

            meal_data,

            colWidths=[
                130,
                320
            ]
        )


        meal_table.setStyle(

            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ])
        )


        story.append(
            meal_table
        )


    story.append(
        Spacer(1, 20)
    )


    story.append(

        Paragraph(

            "<b>Disclaimer:</b> DietAI provides "
            "general wellness and meal-planning "
            "suggestions. Nutrition values are "
            "estimates and are not medical advice.",

            body_style
        )
    )


    document.build(
        story
    )


    buffer.seek(0)


    return buffer


# ============================================================
# SESSION STATE
# ============================================================


defaults = {

    "diet_result": None,

    "chat_history": [],

    "grocery": None,

    "last_replacement": None,

    "food_analysis": None,

    # Kitchen Scanner state
    "kitchen_scan": None,
    "kitchen_confirmed": [],
    "kitchen_recipes": None
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# HEADER
# ============================================================


st.markdown(

    '<div class="main-title">'
    '🥗 DietAI'
    '</div>',

    unsafe_allow_html=True
)


st.markdown(

    '<div class="subtitle">'
    'Your personalized AI nutrition companion'
    '</div>',

    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    st.header(
        "👤 Your Profile"
    )


    name = st.text_input(

        "Name",

        placeholder="Your name"
    )


    age = st.number_input(

        "Age",

        min_value=18,

        max_value=100,

        value=30
    )


    sex = st.selectbox(

        "Sex used for BMR estimate",

        [
            "Male",
            "Female",
            "Prefer not to specify"
        ]
    )


    st.subheader(
        "Height"
    )


    height_unit = st.radio(

        "Height unit",

        [
            "cm",
            "feet/inches"
        ],

        horizontal=True
    )


    if height_unit == "cm":

        height_cm = st.number_input(

            "Height (cm)",

            min_value=100.0,

            max_value=230.0,

            value=170.0
        )

    else:

        feet = st.number_input(

            "Feet",

            min_value=3,

            max_value=8,

            value=5
        )


        inches = st.number_input(

            "Inches",

            min_value=0,

            max_value=11,

            value=7
        )


        height_cm = (

            feet * 30.48

            +

            inches * 2.54
        )


    weight = st.number_input(

        "Weight (kg)",

        min_value=30.0,

        max_value=250.0,

        value=70.0
    )


    st.subheader(
        "🎯 Goal"
    )


    goal = st.selectbox(

        "Your goal",

        [
            "Get Lean",
            "Maintain Weight",
            "Bulk"
        ]
    )


    diet_type = st.selectbox(

        "Diet preference",

        [
            "Vegetarian",
            "Eggetarian",
            "Non-vegetarian"
        ]
    )


    cuisine = st.selectbox(

        "Cuisine",

        [
            "South Indian",
            "North Indian",
            "Any"
        ]
    )


    activity = st.selectbox(

        "Activity",

        [
            "Sedentary",
            "Lightly Active",
            "Moderately Active",
            "Very Active"
        ]
    )


    generate = st.button(

        "🥗 Create My Plan",

        use_container_width=True
    )


# ============================================================
# GENERATE
# ============================================================


if generate:

    if not name.strip():

        st.warning(
            "Please enter your name."
        )

        st.stop()


    initial_state: DietState = {

        "name":
            name,

        "age":
            age,

        "sex":
            sex,

        "height_cm":
            height_cm,

        "weight_kg":
            weight,

        "bmi":
            0,

        "bmi_category":
            "",

        "bmr":
            0,

        "tdee":
            0,

        "calories":
            0,

        "protein_target":
            0,

        "goal":
            goal,

        "diet_type":
            diet_type,

        "cuisine":
            cuisine,

        "activity":
            activity,

        "food_candidates":
            "",

        "validation_attempt":
            0,

        "validation_message":
            "",

        "diet_plan":
            {}
    }


    with st.spinner(

        "AI is creating and validating "
        "your plan..."
    ):

        try:

            result = diet_graph.invoke(
                initial_state
            )


            st.session_state[
                "diet_result"
            ] = result


            st.session_state[
                "chat_history"
            ] = []


            st.session_state[
                "grocery"
            ] = None


            st.session_state[
                "last_replacement"
            ] = None


            st.session_state[
                "food_analysis"
            ] = None

            st.session_state[
                "kitchen_scan"
            ] = None

            st.session_state[
                "kitchen_confirmed"
            ] = []

            st.session_state[
                "kitchen_recipes"
            ] = None


        except Exception as e:

            st.error(
                f"Could not generate plan: {e}"
            )


# ============================================================
# MAIN RESULT
# ============================================================


result = st.session_state[
    "diet_result"
]


if result:

    plan = result[
        "diet_plan"
    ]


    # ========================================================
    # NUTRITION
    # ========================================================


    nutrition = (
        calculate_week_nutrition(
            plan
        )
    )


    weekly_total = (
        nutrition["total"]
    )


    avg_calories = (

        weekly_total["calories"]
        /
        7
    )


    avg_protein = (

        weekly_total["protein"]
        /
        7
    )


    avg_carbs = (

        weekly_total["carbs"]
        /
        7
    )


    avg_fat = (

        weekly_total["fat"]
        /
        7
    )


    # ========================================================
    # DASHBOARD
    # ========================================================


    st.subheader(

        f"Good morning, "
        f"{result['name']} 👋"
    )


    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )


    with col1:

        st.metric(
            "BMI",
            result["bmi"]
        )


    with col2:

        st.metric(
            "BMR",
            f'{result["bmr"]}'
        )


    with col3:

        st.metric(
            "TDEE",
            f'{result["tdee"]}'
        )


    with col4:

        st.metric(

            "Calories Target",

            f'{result["calories"]}'
        )


    with col5:

        st.metric(

            "Protein Target",

            f'{result["protein_target"]}g'
        )


    st.caption(

        "Nutrition values are estimates based "
        "on the current food database."
    )


    # ========================================================
    # AVERAGE NUTRITION
    # ========================================================


    st.subheader(
        "📊 Average Daily Nutrition"
    )


    n1, n2, n3, n4 = (
        st.columns(4)
    )


    with n1:

        st.metric(

            "Calories",

            f"{avg_calories:.0f} kcal"
        )


    with n2:

        st.metric(

            "Protein",

            f"{avg_protein:.0f} g"
        )


    with n3:

        st.metric(

            "Carbs",

            f"{avg_carbs:.0f} g"
        )


    with n4:

        st.metric(

            "Fat",

            f"{avg_fat:.0f} g"
        )


    # ========================================================
    # TARGET PROGRESS
    # ========================================================


    calorie_progress = min(

        avg_calories
        /
        result["calories"],

        1.0
    )


    protein_progress = min(

        avg_protein
        /
        result["protein_target"],

        1.0
    )


    st.markdown(
        "### 🎯 Target Progress"
    )


    st.write(

        f"Calories: "
        f"{avg_calories:.0f} / "
        f"{result['calories']} kcal"
    )


    st.progress(
        calorie_progress
    )


    st.write(

        f"Protein: "
        f"{avg_protein:.0f} / "
        f"{result['protein_target']} g"
    )


    st.progress(
        protein_progress
    )


    # ========================================================
    # TABS
    # ========================================================


    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = (
        st.tabs(

            [
                "📅 My Plan",

                "📊 Nutrition",

                "📸 Food Scanner",

                "🍳 Scan Kitchen",

                "🤖 AI Assistant",

                "🔄 Replace Meal",

                "🛒 Grocery & PDF"
            ]
        )
    )


    # ========================================================
    # TAB 1 - MY PLAN
    # ========================================================


    with tab1:

        st.header(
            "📅 Your 7-Day Plan"
        )


        st.write(
            plan["summary"]
        )


        st.subheader(
            "💡 Recommendations"
        )


        for recommendation in plan[
            "recommendations"
        ]:

            st.write(
                f"• {recommendation}"
            )


        st.divider()


        for day in plan[
            "weekly_plan"
        ]:

            day_nutrition = (
                calculate_day_nutrition(
                    day
                )
            )


            score = nutrition_score(

                day_nutrition["total"],

                result["calories"],

                result["protein_target"],

                day
            )


            with st.expander(

                f"🍽️ {day['day']} "
                f"— Nutrition Score: "
                f"{score}/100",

                expanded=True
            ):

                c1, c2 = (
                    st.columns(2)
                )


                with c1:

                    st.markdown(
                        "### 🌅 Breakfast"
                    )

                    st.write(
                        day["breakfast"]
                    )


                    st.markdown(
                        "### 🍎 Mid Morning"
                    )

                    st.write(
                        day["mid_morning"]
                    )


                    st.markdown(
                        "### 🍛 Lunch"
                    )

                    st.write(
                        day["lunch"]
                    )


                with c2:

                    st.markdown(
                        "### ☕ Evening Snack"
                    )

                    st.write(
                        day["evening_snack"]
                    )


                    st.markdown(
                        "### 🌙 Dinner"
                    )

                    st.write(
                        day["dinner"]
                    )


                st.divider()


                m1, m2, m3, m4 = (
                    st.columns(4)
                )


                with m1:

                    st.metric(

                        "Calories",

                        f'{day_nutrition["total"]["calories"]} kcal'
                    )


                with m2:

                    st.metric(

                        "Protein",

                        f'{day_nutrition["total"]["protein"]} g'
                    )


                with m3:

                    st.metric(

                        "Carbs",

                        f'{day_nutrition["total"]["carbs"]} g'
                    )


                with m4:

                    st.metric(

                        "Fat",

                        f'{day_nutrition["total"]["fat"]} g'
                    )


    # ========================================================
    # TAB 2 - NUTRITION
    # ========================================================


    with tab2:

        st.header(
            "📊 Nutrition Intelligence"
        )


        rows = []


        for day in nutrition[
            "days"
        ]:

            matching_day = next(

                d

                for d in plan[
                    "weekly_plan"
                ]

                if d["day"] == day["day"]
            )


            score = nutrition_score(

                day,

                result["calories"],

                result["protein_target"],

                matching_day
            )


            rows.append({

                "Day":
                    day["day"],

                "Calories":
                    day["calories"],

                "Protein (g)":
                    day["protein"],

                "Carbs (g)":
                    day["carbs"],

                "Fat (g)":
                    day["fat"],

                "Score":
                    score
            })


        st.dataframe(

            rows,

            use_container_width=True,

            hide_index=True
        )


        st.subheader(
            "🎯 Weekly Average"
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        with c1:

            st.metric(
                "Calories",
                f"{avg_calories:.0f}"
            )


        with c2:

            st.metric(
                "Protein",
                f"{avg_protein:.0f} g"
            )


        with c3:

            st.metric(
                "Carbs",
                f"{avg_carbs:.0f} g"
            )


        with c4:

            st.metric(
                "Fat",
                f"{avg_fat:.0f} g"
            )


    # ========================================================
    # TAB 3 - FOOD SCANNER
    # ========================================================


    with tab3:

        st.markdown(
            '<div class="scanner-title">'
            '📸 Food Scanner'
            '</div>',
            unsafe_allow_html=True
        )


        st.write(

            "Upload a photo of your meal, "
            "tell us approximately how many "
            "grams you ate, and DietAI will "
            "estimate the nutrition."
        )


        st.warning(

            "⚠️ Image-based nutrition is an "
            "estimate. A photo cannot reliably "
            "measure exact ingredient quantities, "
            "oil, recipe composition or micronutrients."
        )


        # ----------------------------------------------------
        # UPLOAD
        # ----------------------------------------------------


        uploaded_file = st.file_uploader(

            "Upload your meal photo",

            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],

            key="food_image"
        )


        # ----------------------------------------------------
        # WEIGHT
        # ----------------------------------------------------


        food_weight = st.number_input(

            "Approximate total food weight (grams)",

            min_value=1.0,

            max_value=5000.0,

            value=300.0,

            step=10.0,

            key="food_weight"
        )


        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------


        if uploaded_file:

            st.image(

                uploaded_file,

                caption="Your meal",

                use_container_width=True
            )


        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------


        if st.button(

            "🔍 Analyze My Food",

            type="primary",

            use_container_width=True,

            key="analyze_food"
        ):

            if not uploaded_file:

                st.warning(
                    "Please upload a food image first."
                )

            else:

                image_bytes = (
                    uploaded_file
                    .getvalue()
                )


                mime_type = (
                    uploaded_file.type
                    or
                    "image/jpeg"
                )


                with st.spinner(

                    "AI is looking at your food..."
                ):

                    try:

                        analysis = (
                            analyze_food_image(

                                image_bytes,

                                mime_type,

                                food_weight
                            )
                        )


                        st.session_state[
                            "food_analysis"
                        ] = analysis


                    except Exception as e:

                        st.error(

                            "Food analysis failed. "
                            f"Details: {e}"
                        )


        # ----------------------------------------------------
        # SHOW ANALYSIS
        # ----------------------------------------------------


        analysis = (
            st.session_state[
                "food_analysis"
            ]
        )


        if analysis:

            st.divider()


            st.subheader(
                "🍽️ Food Identified"
            )


            st.markdown(

                f"### {analysis['food_name']}"
            )


            st.write(
                analysis["description"]
            )


            st.info(

                f"Estimated serving weight: "
                f"{analysis['estimated_weight_g']:.0f} g"
            )


            # =================================================
            # MACROS
            # =================================================


            st.subheader(
                "🔥 Calories & Macronutrients"
            )


            c1, c2, c3, c4 = (
                st.columns(4)
            )


            with c1:

                st.metric(

                    "Calories",

                    f'{analysis["calories"]:.0f} kcal'
                )


            with c2:

                st.metric(

                    "Protein",

                    f'{analysis["protein_g"]:.1f} g'
                )


            with c3:

                st.metric(

                    "Carbohydrates",

                    f'{analysis["carbohydrates_g"]:.1f} g'
                )


            with c4:

                st.metric(

                    "Fat",

                    f'{analysis["fat_g"]:.1f} g'
                )


            # =================================================
            # OTHER
            # =================================================


            st.subheader(
                "🥗 Other Nutrition"
            )


            c1, c2, c3, c4 = (
                st.columns(4)
            )


            with c1:

                st.metric(

                    "Fiber",

                    f'{analysis["fiber_g"]:.1f} g'
                )


            with c2:

                st.metric(

                    "Sugar",

                    f'{analysis["sugar_g"]:.1f} g'
                )


            with c3:

                st.metric(

                    "Saturated Fat",

                    f'{analysis["saturated_fat_g"]:.1f} g'
                )


            with c4:

                st.metric(

                    "Sodium",

                    f'{analysis["sodium_mg"]:.0f} mg'
                )


            # =================================================
            # MINERALS
            # =================================================


            st.subheader(
                "🧂 Minerals"
            )


            minerals = {

                "Potassium":
                    f'{analysis["potassium_mg"]:.0f} mg',

                "Calcium":
                    f'{analysis["calcium_mg"]:.0f} mg',

                "Iron":
                    f'{analysis["iron_mg"]:.1f} mg',

                "Magnesium":
                    f'{analysis["magnesium_mg"]:.0f} mg',

                "Zinc":
                    f'{analysis["zinc_mg"]:.1f} mg',

                "Phosphorus":
                    f'{analysis["phosphorus_mg"]:.0f} mg'
            }


            mineral_rows = [

                [
                    "Nutrient",
                    "Estimated Amount"
                ]
            ]


            for key, value in (
                minerals.items()
            ):

                mineral_rows.append(
                    [
                        key,
                        value
                    ]
                )


            st.table(
                mineral_rows
            )


            # =================================================
            # VITAMINS
            # =================================================


            st.subheader(
                "💊 Vitamins"
            )


            vitamins = {

                "Vitamin A":
                    f'{analysis["vitamin_a_mcg"]:.0f} µg',

                "Vitamin C":
                    f'{analysis["vitamin_c_mg"]:.1f} mg',

                "Vitamin D":
                    f'{analysis["vitamin_d_mcg"]:.1f} µg',

                "Vitamin E":
                    f'{analysis["vitamin_e_mg"]:.1f} mg',

                "Vitamin K":
                    f'{analysis["vitamin_k_mcg"]:.1f} µg',

                "Vitamin B1":
                    f'{analysis["vitamin_b1_mg"]:.2f} mg',

                "Vitamin B2":
                    f'{analysis["vitamin_b2_mg"]:.2f} mg',

                "Vitamin B3":
                    f'{analysis["vitamin_b3_mg"]:.2f} mg',

                "Vitamin B6":
                    f'{analysis["vitamin_b6_mg"]:.2f} mg',

                "Vitamin B12":
                    f'{analysis["vitamin_b12_mcg"]:.2f} µg',

                "Folate":
                    f'{analysis["folate_mcg"]:.0f} µg'
            }


            vitamin_rows = [

                [
                    "Vitamin",
                    "Estimated Amount"
                ]
            ]


            for key, value in (
                vitamins.items()
            ):

                vitamin_rows.append(
                    [
                        key,
                        value
                    ]
                )


            st.table(
                vitamin_rows
            )


            # =================================================
            # CONFIDENCE
            # =================================================


            st.subheader(
                "🎯 Estimation Confidence"
            )


            st.info(
                analysis["confidence"]
            )


            # =================================================
            # ASSUMPTIONS
            # =================================================


            st.subheader(
                "📝 Assumptions"
            )


            for assumption in (
                analysis["assumptions"]
            ):

                st.write(
                    f"• {assumption}"
                )


            # =================================================
            # ADD TO DAILY INTAKE
            # =================================================


            st.divider()


            st.subheader(
                "➕ Add to Today's Intake"
            )


            if st.button(

                "Add This Food to Today's Intake",

                use_container_width=True,

                key="add_scanned_food"
            ):

                st.session_state[
                    "today_scanned_food"
                ] = analysis


                st.success(

                    "Food added to today's "
                    "scanned intake."
                )


            # ------------------------------------------------
            # TODAY SCANNED FOOD
            # ------------------------------------------------


            if (
                "today_scanned_food"
                in st.session_state
            ):

                scanned = (
                    st.session_state[
                        "today_scanned_food"
                    ]
                )


                st.markdown(
                    "### Today's Scanned Food"
                )


                sc1, sc2, sc3 = (
                    st.columns(3)
                )


                with sc1:

                    st.metric(

                        "Calories",

                        f'{scanned["calories"]:.0f} kcal'
                    )


                with sc2:

                    st.metric(

                        "Protein",

                        f'{scanned["protein_g"]:.1f} g'
                    )


                with sc3:

                    st.metric(

                        "Carbs",

                        f'{scanned["carbohydrates_g"]:.1f} g'
                    )



    # ========================================================
    # TAB 4 - SCAN KITCHEN
    # ========================================================

    with tab4:

        st.header("🍳 What's in My Kitchen?")

        st.write(
            "Upload photos of your refrigerator, pantry or kitchen shelf. "
            "DietAI will identify the ingredients it can see and create "
            "recipes using what you already have."
        )

        st.info(
            "💡 Tip: upload multiple photos for better coverage — "
            "for example, refrigerator + pantry + vegetables."
        )

        kitchen_images = st.file_uploader(
            "Upload kitchen photos",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="kitchen_images"
        )

        if kitchen_images:
            st.subheader("📷 Kitchen Photos")
            cols = st.columns(min(len(kitchen_images), 3))

            for i, image in enumerate(kitchen_images):
                with cols[i % len(cols)]:
                    st.image(
                        image,
                        caption=image.name,
                        use_container_width=True
                    )

        if st.button(
            "🔍 Scan My Kitchen",
            type="primary",
            use_container_width=True,
            key="scan_kitchen_button"
        ):
            if not kitchen_images:
                st.warning("Please upload at least one kitchen photo.")
            else:
                with st.spinner("AI is scanning your kitchen..."):
                    try:
                        scan_result = scan_kitchen_images(kitchen_images)

                        st.session_state["kitchen_scan"] = scan_result
                        st.session_state["kitchen_confirmed"] = [
                            item["name"]
                            for item in scan_result["ingredients"]
                        ]
                        st.session_state["kitchen_recipes"] = None

                        st.success("Kitchen scanned successfully!")
                    except Exception as e:
                        st.error(f"Kitchen scan failed: {e}")

        kitchen_scan = st.session_state["kitchen_scan"]

        if kitchen_scan:
            st.divider()
            st.subheader("🧺 Ingredients Detected")

            st.caption(
                "Review the detected items. Remove anything incorrect "
                "and add anything the AI missed."
            )

            selected_ingredients = []

            for i, item in enumerate(kitchen_scan["ingredients"]):
                item_name = item["name"]

                selected = st.checkbox(
                    f'{item_name} · {item["category"]} · '
                    f'{item["confidence"]} confidence',
                    value=(
                        item_name
                        in st.session_state["kitchen_confirmed"]
                    ),
                    key=f"kitchen_item_{i}"
                )

                if selected:
                    selected_ingredients.append(item_name)

            st.subheader("➕ Add Something AI Missed")

            manual_items = st.text_input(
                "Add ingredients",
                placeholder="Example: paneer, tomato, green chilli",
                key="manual_kitchen_items"
            )

            if manual_items.strip():
                for item in manual_items.split(","):
                    item = item.strip()
                    if item and item not in selected_ingredients:
                        selected_ingredients.append(item)

            st.write(
                f"**Confirmed ingredients:** {len(selected_ingredients)}"
            )

            cooking_time = st.selectbox(
                "Maximum cooking time",
                [15, 30, 45, 60],
                format_func=lambda x: f"{x} minutes",
                key="kitchen_cooking_time"
            )

            preference = st.text_input(
                "Recipe preference",
                placeholder=(
                    "Example: high protein, breakfast, less oil, quick dinner"
                ),
                key="kitchen_recipe_preference"
            )

            if st.button(
                "👨‍🍳 Create Recipes From My Kitchen",
                type="primary",
                use_container_width=True,
                key="create_kitchen_recipes"
            ):
                if not selected_ingredients:
                    st.warning("Please select at least one ingredient.")
                else:
                    with st.spinner(
                        "DietAI is creating recipes from your kitchen..."
                    ):
                        try:
                            recipe_result = generate_kitchen_recipes(
                                selected_ingredients,
                                result,
                                cooking_time,
                                preference
                            )
                            st.session_state["kitchen_recipes"] = recipe_result
                        except Exception as e:
                            st.error(f"Recipe generation failed: {e}")

        kitchen_recipes = st.session_state["kitchen_recipes"]

        if kitchen_recipes:
            st.divider()
            st.subheader("🍽️ Recipes You Can Make")

            for i, recipe in enumerate(kitchen_recipes["recipes"]):
                with st.expander(
                    f'🍳 {i + 1}. {recipe["name"]}',
                    expanded=(i == 0)
                ):
                    st.write(recipe["description"])

                    c1, c2, c3, c4, c5 = st.columns(5)

                    with c1:
                        st.metric(
                            "Calories",
                            f'{recipe["calories_per_serving"]} kcal'
                        )
                    with c2:
                        st.metric(
                            "Protein",
                            f'{recipe["protein_g"]:.1f} g'
                        )
                    with c3:
                        st.metric(
                            "Carbs",
                            f'{recipe["carbohydrates_g"]:.1f} g'
                        )
                    with c4:
                        st.metric(
                            "Fat",
                            f'{recipe["fat_g"]:.1f} g'
                        )
                    with c5:
                        st.metric(
                            "Fiber",
                            f'{recipe["fiber_g"]:.1f} g'
                        )

                    st.write(
                        f'⏱️ **Cooking time:** '
                        f'{recipe["cooking_time_minutes"]} minutes'
                    )
                    st.write(
                        f'🍽️ **Servings:** {recipe["servings"]}'
                    )

                    st.markdown("### 🥕 Ingredients")
                    for ingredient in recipe["ingredients"]:
                        st.write(f"• {ingredient}")

                    if recipe["missing_ingredients"]:
                        st.markdown("### 🛒 Missing Ingredients")
                        for missing in recipe["missing_ingredients"]:
                            st.write(f"• {missing}")

                    st.markdown("### 👨‍🍳 Instructions")
                    for step_no, step in enumerate(
                        recipe["instructions"],
                        start=1
                    ):
                        st.write(f"**Step {step_no}:** {step}")

    # ========================================================
    # TAB 5 - AI ASSISTANT
    # ========================================================


    with tab5:

        st.header(
            "🤖 Ask DietAI"
        )


        st.caption(

            "Ask questions about your "
            "current plan."
        )


        for message in (
            st.session_state[
                "chat_history"
            ]
        ):

            with st.chat_message(
                message["role"]
            ):

                st.write(
                    message["content"]
                )


        question = st.chat_input(

            "Ask something about your diet..."
        )


        if question:

            st.session_state[
                "chat_history"
            ].append({

                "role":
                    "user",

                "content":
                    question
            })


            conversation = "\n".join(

                [

                    f'{m["role"]}: '
                    f'{m["content"]}'

                    for m in (
                        st.session_state[
                            "chat_history"
                        ]
                    )
                ]
            )


            prompt = f"""

You are DietAI.

Help the user understand their
existing nutrition plan.

USER PROFILE
============

Name:
{result["name"]}

Age:
{result["age"]}

Weight:
{result["weight_kg"]} kg

Height:
{result["height_cm"]} cm

BMI:
{result["bmi"]}

Goal:
{result["goal"]}

Diet:
{result["diet_type"]}

Cuisine:
{result["cuisine"]}

Daily calorie target:
{result["calories"]}

Protein target:
{result["protein_target"]} g


CURRENT PLAN
============

{json.dumps(
    plan,
    indent=2
)}


CURRENT NUTRITION
=================

{json.dumps(
    nutrition,
    indent=2
)}


CONVERSATION
============

{conversation}


Answer clearly and practically.

Do not provide medical diagnosis.
"""


            with st.spinner(
                "Thinking..."
            ):

                response = llm.invoke(
                    prompt
                )


            st.session_state[
                "chat_history"
            ].append({

                "role":
                    "assistant",

                "content":
                    response.content
            })


            st.rerun()


    # ========================================================
    # TAB 6 - REPLACE MEAL
    # ========================================================


    with tab6:

        st.header(
            "🔄 Replace a Meal"
        )


        days = [

            day["day"]

            for day in plan[
                "weekly_plan"
            ]
        ]


        selected_day = st.selectbox(

            "Select day",

            days,

            key="replace_day"
        )


        meal_map = {

            "Breakfast":
                "breakfast",

            "Mid Morning":
                "mid_morning",

            "Lunch":
                "lunch",

            "Evening Snack":
                "evening_snack",

            "Dinner":
                "dinner"
        }


        selected_meal_display = (
            st.selectbox(

                "Select meal",

                list(
                    meal_map.keys()
                ),

                key="replace_meal"
            )
        )


        selected_meal = meal_map[
            selected_meal_display
        ]


        selected_day_data = None


        for day in plan[
            "weekly_plan"
        ]:

            if day["day"] == selected_day:

                selected_day_data = day

                break


        current_meal = (
            selected_day_data[
                selected_meal
            ]
        )


        st.markdown(
            "### Current Meal"
        )


        st.info(
            current_meal
        )


        current_nutrition = (
            calculate_meal_nutrition(
                current_meal
            )
        )


        st.caption(

            f'Current nutrition: '
            f'{current_nutrition["calories"]} kcal | '
            f'{current_nutrition["protein"]}g protein'
        )


        reason = st.selectbox(

            "Why replace it?",

            [
                "I don't like it",

                "It is not available",

                "I want more protein",

                "I want fewer calories",

                "I want a different food"
            ],

            key="replacement_reason"
        )


        if st.button(

            "🔄 Find Replacement",

            use_container_width=True,

            key="replace_button"
        ):

            compatible_foods = []


            for food in FOODS:

                if (
                    result["diet_type"]
                    ==
                    "Vegetarian"
                ):

                    if food["diet"] != "Vegetarian":

                        continue


                elif (
                    result["diet_type"]
                    ==
                    "Eggetarian"
                ):

                    if food["diet"] not in [

                        "Vegetarian",

                        "Eggetarian"
                    ]:

                        continue


                if (
                    result["cuisine"]
                    !=
                    "Any"
                ):

                    if food["cuisine"] not in [

                        result["cuisine"],

                        "Indian"
                    ]:

                        continue


                compatible_foods.append(
                    food
                )


            candidate_text = "\n".join(

                [

                    (
                        f'ID={food["id"]}; '
                        f'Name={food["name"]}; '
                        f'Serving={food["serving"]}; '
                        f'Calories={food["calories"]}; '
                        f'Protein={food["protein"]}g'
                    )

                    for food
                    in compatible_foods
                ]
            )


            replacement_prompt = f"""

You are DietAI.

Choose ONE replacement food from
the available food database.

USER

Goal:
{result["goal"]}

Diet:
{result["diet_type"]}

Cuisine:
{result["cuisine"]}

Daily calories:
{result["calories"]}

Protein target:
{result["protein_target"]} g


CURRENT MEAL

{current_meal}

Current calories:
{current_nutrition["calories"]}

Current protein:
{current_nutrition["protein"]}


REASON

{reason}


AVAILABLE FOODS

{candidate_text}


RULES

Choose exactly one food ID.

Respect the user's diet.

Respect the user's cuisine.

Prefer similar calories.

If more protein is requested,
choose higher protein.

If fewer calories are requested,
choose lower calories.

Return the exact food ID.
"""


            with st.spinner(
                "Finding replacement..."
            ):

                replacement = (
                    structured_replacement_llm
                    .invoke(
                        replacement_prompt
                    )
                )


            food_id = (
                replacement[
                    "replacement_food_id"
                ]
                if isinstance(
                    replacement,
                    dict
                )
                else
                replacement.replacement_food_id
            )


            if food_id not in FOOD_LOOKUP:

                st.error(
                    "AI returned an invalid food. "
                    "Please try again."
                )

            else:

                new_food = (
                    FOOD_LOOKUP[
                        food_id
                    ]
                )


                new_meal = (

                    f'{new_food["name"]} '
                    f'({new_food["serving"]})'
                )


                # Update actual plan

                for day in plan[
                    "weekly_plan"
                ]:

                    if day["day"] == selected_day:

                        day[
                            selected_meal
                        ] = new_meal

                        break


                result[
                    "diet_plan"
                ] = plan


                st.session_state[
                    "diet_result"
                ] = result


                st.session_state[
                    "last_replacement"
                ] = {

                    "day":
                        selected_day,

                    "meal":
                        selected_meal_display,

                    "old":
                        current_meal,

                    "new":
                        new_meal,

                    "old_calories":
                        current_nutrition[
                            "calories"
                        ],

                    "new_calories":
                        new_food[
                            "calories"
                        ],

                    "old_protein":
                        current_nutrition[
                            "protein"
                        ],

                    "new_protein":
                        new_food[
                            "protein"
                        ]
                }


                st.success(
                    "Meal successfully replaced!"
                )


                st.write(

                    f"**Previous:** "
                    f"{current_meal}"
                )


                st.write(

                    f"**New:** "
                    f"{new_meal}"
                )


                st.write(

                    f'Calories: '
                    f'{current_nutrition["calories"]} → '
                    f'{new_food["calories"]} kcal'
                )


                st.write(

                    f'Protein: '
                    f'{current_nutrition["protein"]} → '
                    f'{new_food["protein"]} g'
                )


                st.rerun()


        if (
            st.session_state[
                "last_replacement"
            ]
            is not None
        ):

            replacement = (
                st.session_state[
                    "last_replacement"
                ]
            )


            st.success(

                f'Replaced '
                f'{replacement["meal"]} '
                f'on '
                f'{replacement["day"]}.'
            )


    # ========================================================
    # TAB 7 - GROCERY + PDF
    # ========================================================


    with tab7:

        st.header(
            "🛒 Grocery List"
        )


        if st.button(

            "🛒 Generate Grocery List",

            key="grocery_button"
        ):

            grocery_prompt = f"""

Create a practical grocery list
for this 7-day meal plan.

Diet:
{result["diet_type"]}

Cuisine:
{result["cuisine"]}

Plan:

{json.dumps(
    plan,
    indent=2
)}

Group items into:

Grains
Proteins
Vegetables
Fruits
Dairy
Other

Keep it practical.
"""


            with st.spinner(
                "Creating grocery list..."
            ):

                grocery = (
                    structured_grocery_llm
                    .invoke(
                        grocery_prompt
                    )
                )


            st.session_state[
                "grocery"
            ] = grocery.model_dump()


        if (
            st.session_state[
                "grocery"
            ]
            is not None
        ):

            grocery = (
                st.session_state[
                    "grocery"
                ]
            )


            c1, c2 = (
                st.columns(2)
            )


            with c1:

                st.subheader(
                    "🌾 Grains"
                )

                for item in grocery[
                    "grains"
                ]:

                    st.write(
                        f"☐ {item}"
                    )


                st.subheader(
                    "🥩 Proteins"
                )

                for item in grocery[
                    "proteins"
                ]:

                    st.write(
                        f"☐ {item}"
                    )


                st.subheader(
                    "🥛 Dairy"
                )

                for item in grocery[
                    "dairy"
                ]:

                    st.write(
                        f"☐ {item}"
                    )


            with c2:

                st.subheader(
                    "🥬 Vegetables"
                )

                for item in grocery[
                    "vegetables"
                ]:

                    st.write(
                        f"☐ {item}"
                    )


                st.subheader(
                    "🍎 Fruits"
                )

                for item in grocery[
                    "fruits"
                ]:

                    st.write(
                        f"☐ {item}"
                    )


                st.subheader(
                    "📦 Other"
                )

                for item in grocery[
                    "other"
                ]:

                    st.write(
                        f"☐ {item}"
                    )


        # ====================================================
        # PDF
        # ====================================================


        st.divider()


        st.header(
            "📥 Download"
        )


        latest_nutrition = (
            calculate_week_nutrition(
                plan
            )
        )


        try:

            pdf = create_pdf(

                result,

                latest_nutrition
            )


            st.download_button(

                label=
                "📄 Download Updated Diet Plan PDF",

                data=pdf,

                file_name=(
                    f'{result["name"]}'
                    f'_diet_plan.pdf'
                ),

                mime=
                "application/pdf",

                use_container_width=True,

                key="pdf_download"
            )


        except Exception as e:

            st.error(

                f"PDF generation failed: "
                f"{e}"
            )


# ============================================================
# FOOTER
# ============================================================


st.divider()


st.caption(

    "DietAI provides general wellness and "
    "meal-planning suggestions. BMI, calorie "
    "and nutrition values are estimates. "
    "The Food Scanner is an image-based estimate "
    "and should not be treated as laboratory "
    "nutrition analysis or medical advice."
)