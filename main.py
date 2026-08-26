import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent


# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ============================================================
# WEB SEARCH TOOL
# ============================================================

search = DuckDuckGoSearchRun()


@tool
def recipe_search(dish: str) -> str:
    """
    Search the web for recipe ingredients and cooking instructions.
    """

    try:
        result = search.invoke(
            f"{dish} recipe ingredients cooking instructions"
        )

        return result[:4000]

    except Exception as e:
        return f"Recipe search failed: {str(e)}"


# ============================================================
# GROCERY LIST TOOL
# ============================================================

@tool
def grocery_list(dish: str, servings: int) -> str:
    """
    Calculate ingredient quantities for a dish and number of servings.
    """

    recipes = {

        "chicken biryani": {
            "chicken": 1.0,
            "basmati rice": 0.75,
            "onion": 0.5,
            "tomato": 0.25,
            "yogurt": 0.25,
            "ginger": 0.05,
            "garlic": 0.05,
            "biryani masala": 1,
            "cooking oil": 0.1
        },

        "veg biryani": {
            "basmati rice": 0.75,
            "mixed vegetables": 0.75,
            "onion": 0.5,
            "tomato": 0.25,
            "yogurt": 0.25,
            "ginger": 0.05,
            "garlic": 0.05,
            "biryani masala": 1,
            "cooking oil": 0.1
        },

        "pasta": {
            "pasta": 0.5,
            "tomato sauce": 0.5,
            "onion": 0.25,
            "garlic": 0.03,
            "cheese": 0.2,
            "cooking oil": 0.05
        },

        "chicken curry": {
            "chicken": 1.0,
            "onion": 0.5,
            "tomato": 0.5,
            "ginger": 0.05,
            "garlic": 0.05,
            "cooking oil": 0.1,
            "spices": 1
        }
    }

    dish_key = dish.lower().strip()

    if dish_key not in recipes:
        return (
            f"No predefined grocery template found for {dish}.\n"
            f"Supported dishes: {', '.join(recipes.keys())}"
        )

    if servings <= 0:
        return "Servings must be greater than 0."

    # Recipes are based on 5 servings
    multiplier = servings / 5

    lines = [
        f"🛒 Grocery List for {dish.title()}",
        f"👥 Servings: {servings}",
        ""
    ]

    for item, quantity in recipes[dish_key].items():

        scaled_quantity = quantity * multiplier

        if "masala" in item or "spices" in item:
            unit = "packet"
        elif item == "cooking oil":
            unit = "litre"
        else:
            unit = "kg"

        lines.append(
            f"- {item.title()}: {scaled_quantity:.2f} {unit}"
        )

    return "\n".join(lines)


# ============================================================
# COST CALCULATOR TOOL
# ============================================================

@tool
def cost_calculator(dish: str, servings: int) -> str:
    """
    Estimate the total grocery cost for a dish and number of servings.
    """

    prices = {

        "chicken biryani": 550,
        "veg biryani": 350,
        "pasta": 300,
        "chicken curry": 500
    }

    dish_key = dish.lower().strip()

    if dish_key not in prices:
        return (
            f"No cost estimate available for {dish}.\n"
            f"Supported dishes: {', '.join(prices.keys())}"
        )

    if servings <= 0:
        return "Servings must be greater than 0."

    estimated_cost = prices[dish_key] * (servings / 5)

    return (
        f"💰 Estimated Cost\n"
        f"🍽️ Dish: {dish.title()}\n"
        f"👥 Servings: {servings}\n"
        f"💵 Estimated Total: ₹{estimated_cost:,.0f}"
    )


# ============================================================
# AGENT
# ============================================================

tools = [
    recipe_search,
    grocery_list,
    cost_calculator
]


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are SmartRecipe Agent.

Your job is to help users with:

1. Recipe searches
2. Grocery lists
3. Grocery cost estimates

Rules:

- If the user asks for a recipe, use the recipe_search tool.
- If the user asks for a grocery list, use the grocery_list tool.
- If the user asks for a cost or budget estimate, use the cost_calculator tool.
- If the user asks for all three, use all relevant tools.
- Pay attention to the number of servings.
- Give clear and simple answers.
- Clearly mention when a cost is only an estimate.
- Do not invent grocery quantities when the grocery_list tool does not support the requested dish.
"""
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SmartRecipe Agent",
    description="AI Recipe, Grocery List and Cost Estimation Agent",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>SmartRecipe Agent</title>

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <style>

            * {
                box-sizing: border-box;
            }

            body {
                font-family: Arial, sans-serif;
                background: #f7f7f7;
                margin: 0;
                padding: 0;
            }

            .container {
                max-width: 750px;
                margin: 50px auto;
                padding: 25px;
            }

            .card {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.08);
            }

            h1 {
                text-align: center;
                color: #d9480f;
                margin-bottom: 10px;
            }

            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 25px;
            }

            textarea {
                width: 100%;
                height: 130px;
                padding: 15px;
                font-size: 16px;
                border-radius: 10px;
                border: 1px solid #ccc;
                resize: vertical;
                outline: none;
            }

            textarea:focus {
                border-color: #d9480f;
            }

            button {
                width: 100%;
                background: #d9480f;
                color: white;
                border: none;
                padding: 14px;
                margin-top: 15px;
                border-radius: 10px;
                font-size: 16px;
                cursor: pointer;
            }

            button:hover {
                background: #b93808;
            }

            button:disabled {
                background: #aaa;
                cursor: not-allowed;
            }

            #result {
                margin-top: 25px;
                padding: 20px;
                background: #fafafa;
                border: 1px solid #e5e5e5;
                border-radius: 10px;
                white-space: pre-wrap;
                line-height: 1.6;
                min-height: 100px;
            }

            .examples {
                margin-top: 20px;
                color: #555;
            }

            .examples p {
                margin: 8px 0;
            }

        </style>

    </head>


    <body>

        <div class="container">

            <div class="card">

                <h1>🍳 SmartRecipe Agent</h1>

                <p class="subtitle">
                    Recipe Search • Grocery List • Cost Calculator
                </p>


                <textarea
                    id="msg"
                    placeholder="Example: Find me a chicken biryani recipe for 10 people, make a grocery list and calculate the total cost."
                ></textarea>


                <button
                    id="button"
                    onclick="send()"
                >
                    Run Agent
                </button>


                <div id="result">
                    Response output will appear here...
                </div>


                <div class="examples">

                    <strong>Try these:</strong>

                    <p>
                        🍗 Chicken biryani for 10 people
                    </p>

                    <p>
                        🥕 Make a grocery list for veg biryani for 5 people
                    </p>

                    <p>
                        💰 What is the cost of pasta for 10 people?
                    </p>

                </div>

            </div>

        </div>


        <script>

            async function send() {

                const text =
                    document.getElementById("msg").value.trim();

                const out =
                    document.getElementById("result");

                const button =
                    document.getElementById("button");


                if (!text) {

                    out.innerText =
                        "⚠️ Please enter a question.";

                    return;
                }


                button.disabled = true;

                button.innerText =
                    "⏳ Running Agent...";


                out.innerText =
                    "🤖 Thinking and invoking tools...";


                try {

                    const res = await fetch(
                        "/chat",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body: JSON.stringify({
                                message: text
                            })
                        }
                    );


                    const data =
                        await res.json();


                    if (!res.ok) {

                        out.innerText =
                            "❌ Error: " +
                            (data.detail ||
                             "Something went wrong.");

                    } else {

                        out.innerText =
                            data.response;

                    }

                }

                catch (error) {

                    out.innerText =
                        "❌ Connection error: " +
                        error.message;

                }

                finally {

                    button.disabled = false;

                    button.innerText =
                        "Run Agent";

                }

            }

        </script>

    </body>

    </html>
    """


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(req: ChatRequest):

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": req.message
                    }
                ]
            }
        )

        final_message = result["messages"][-1]

        return {
            "response": final_message.content
        }

    except Exception as e:

        return {
            "response": f"❌ Agent Error: {str(e)}"
        }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "SmartRecipe Agent"
    }
