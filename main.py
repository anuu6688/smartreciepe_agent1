import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
search = DuckDuckGoSearchRun()

@tool
def recipe_search(dish: str) -> str:
    """Search web for recipe instructions."""
    try:
        return search.invoke(f"{dish} recipe ingredients cooking instructions")[:4000]
    except Exception as e:
        return f"Recipe search failed: {str(e)}"

@tool
def grocery_list(dish: str, servings: int) -> str:
    """Calculate ingredient quantities for a dish and servings."""
    recipes = {
        "chicken biryani": {"chicken": 1.0, "basmati rice": 0.75, "onion": 0.5, "tomato": 0.25, "yogurt": 0.25, "ginger": 0.05, "garlic": 0.05, "biryani masala": 1, "cooking oil": 0.1},
        "veg biryani": {"basmati rice": 0.75, "mixed vegetables": 0.75, "onion": 0.5, "tomato": 0.25, "yogurt": 0.25, "ginger": 0.05, "garlic": 0.05, "biryani masala": 1, "cooking oil": 0.1},
        "pasta": {"pasta": 0.5, "tomato sauce": 0.5, "onion": 0.25, "garlic": 0.03, "cheese": 0.2, "cooking oil": 0.05},
        "chicken curry": {"chicken": 1.0, "onion": 0.5, "tomato": 0.5, "ginger": 0.05, "garlic": 0.05, "cooking oil": 0.1, "spices": 1}
    }
    dish_key = dish.lower().strip()
    if dish_key not in recipes:
        return f"No predefined template found for {dish}."
    mult = servings / 5
    lines = [f"🛒 Grocery List for {dish.title()} ({servings} servings):"]
    for item, qty in recipes[dish_key].items():
        scaled = qty * mult
        unit = "packet" if "masala" in item or "spices" in item else "kg/litre"
        lines.append(f"- {item.title()}: {scaled:.2f} {unit}")
    return "\n".join(lines)

@tool
def cost_calculator(dish: str, servings: int) -> str:
    """Estimate total grocery costs."""
    prices = {"chicken biryani": 550, "veg biryani": 350, "pasta": 300, "chicken curry": 500}
    dish_key = dish.lower().strip()
    if dish_key not in prices:
        return f"No cost estimate available for {dish}."
    return f"💰 Estimated Cost for {dish.title()} ({servings} servings): ₹{prices[dish_key] * (servings / 5):,.0f}"

tools = [recipe_search, grocery_list, cost_calculator]
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are SmartRecipe Agent. Coordinate recipes, grocery lists, and costs accurately."
)

app = FastAPI(title="SmartRecipe Agent")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartRecipe Agent</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; background: #fafafa; color: #222; }
            h1 { color: #d9480f; }
            textarea { width: 100%; height: 100px; padding: 12px; font-size: 15px; border-radius: 8px; border: 1px solid #ccc; box-sizing: border-box; }
            button { background: #d9480f; color: white; border: none; padding: 12px 20px; border-radius: 6px; font-size: 15px; cursor: pointer; margin-top: 10px; }
            button:hover { background: #b83807; }
            #result { margin-top: 25px; padding: 20px; background: #fff; border: 1px solid #e9ecef; border-radius: 8px; white-space: pre-wrap; line-height: 1.6; }
        </style>
    </head>
    <body>
        <h1>🍳 SmartRecipe Agent</h1>
        <p>Ask for recipes, grocery ingredient lists, and approximate budget estimates.</p>
        <textarea id="msg" placeholder="e.g. Find me a recipe for chicken biryani for 10 people with grocery list and total cost."></textarea><br>
        <button onclick="send()">Run Agent</button>
        <div id="result">Response output will appear here...</div>
        <script>
            async function send() {
                const text = document.getElementById("msg").value;
                const out = document.getElementById("result");
                out.innerText = "⏳ Thinking and invoking tools...";
                const res = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: text})
                });
                const data = await res.json();
                out.innerText = data.response;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
def chat(req: ChatRequest):
    result = agent.invoke({"messages": [("user", req.message)]})
    return {"response": result["messages"][-1].content}
