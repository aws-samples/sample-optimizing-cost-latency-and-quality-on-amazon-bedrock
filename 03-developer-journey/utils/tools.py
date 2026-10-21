"""
Tool definitions for the customer support agent.
Adapted from AgentCore E2E sample (lab-01-create-an-agent.ipynb).
"""

from __future__ import annotations

import boto3
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException
from strands.tools import tool


@tool
def get_return_policy(product_category: str) -> str:
    """Get return policy information for a specific product category.

    Args:
        product_category: The category of product (e.g., 'smartphones', 'laptops', 'accessories')

    Returns:
        Detailed return policy information for the specified category
    """
    return_policies = {
        "smartphones": {
            "window": "30 days",
            "condition": "Original packaging, no physical damage, factory reset required",
            "process": "Online RMA portal or technical support",
            "refund_time": "5-7 business days after inspection",
            "shipping": "Free return shipping, prepaid label provided",
            "warranty": "1-year manufacturer warranty included",
            "restocking_fee": "None for unopened, 15% for opened items",
        },
        "laptops": {
            "window": "30 days",
            "condition": "Original packaging, all accessories included, no physical damage",
            "process": "Online RMA portal or in-store return",
            "refund_time": "7-10 business days after inspection",
            "shipping": "Free return shipping for defects, customer pays for change of mind",
            "warranty": "1-year manufacturer warranty, extended warranty available",
            "restocking_fee": "None for defects, 15% for change of mind",
        },
        "accessories": {
            "window": "60 days",
            "condition": "Unopened original packaging preferred, opened items accepted",
            "process": "Online RMA portal, mail-in, or in-store",
            "refund_time": "3-5 business days after inspection",
            "shipping": "Free return shipping on orders over $50",
            "warranty": "90-day manufacturer warranty",
            "restocking_fee": "None",
        },
        "tablets": {
            "window": "30 days",
            "condition": "Original packaging, factory reset, no physical damage",
            "process": "Online RMA portal or technical support",
            "refund_time": "5-7 business days after inspection",
            "shipping": "Free return shipping, prepaid label provided",
            "warranty": "1-year manufacturer warranty included",
            "restocking_fee": "None for unopened, 10% for opened items",
        },
        "audio": {
            "window": "45 days",
            "condition": "Original packaging, all accessories included",
            "process": "Online RMA portal or in-store",
            "refund_time": "5-7 business days after inspection",
            "shipping": "Free return shipping",
            "warranty": "1-year manufacturer warranty",
            "restocking_fee": "None",
        },
    }

    default_policy = {
        "window": "30 days",
        "condition": "Original packaging, undamaged condition",
        "process": "Contact customer support for RMA",
        "refund_time": "7-10 business days after inspection",
        "shipping": "Varies by product",
        "warranty": "Standard manufacturer warranty",
        "restocking_fee": "May apply",
    }

    policy = return_policies.get(product_category.lower(), default_policy)
    category_name = product_category.title()

    return f"""Return Policy - {category_name}:

- Return Window: {policy["window"]}
- Condition Requirements: {policy["condition"]}
- Return Process: {policy["process"]}
- Refund Timeline: {policy["refund_time"]}
- Shipping: {policy["shipping"]}
- Warranty: {policy["warranty"]}
- Restocking Fee: {policy["restocking_fee"]}

For assistance with returns, visit our online RMA portal or contact customer support."""


@tool
def get_product_info(product_type: str) -> str:
    """Get detailed technical specifications for electronics products.

    Args:
        product_type: The type of product (e.g., 'laptops', 'smartphones', 'tablets')

    Returns:
        Detailed product specifications and features
    """
    products = {
        "laptops": {
            "featured_model": "TechMart Pro 15",
            "price": "$1,299",
            "specs": {
                "processor": "Intel Core i7-13700H (14 cores, up to 5.0GHz)",
                "memory": "16GB DDR5 RAM (upgradeable to 32GB)",
                "storage": "512GB NVMe SSD (expandable)",
                "display": '15.6" FHD IPS, 144Hz refresh rate, 300 nits',
                "graphics": "NVIDIA RTX 4060 6GB",
                "battery": "72Wh, up to 10 hours",
                "weight": "4.5 lbs (2.04 kg)",
                "ports": "2x USB-C, 2x USB-A, HDMI 2.1, SD card reader",
            },
            "colors": ["Space Gray", "Silver", "Midnight Blue"],
            "in_stock": True,
            "delivery": "2-3 business days",
        },
        "smartphones": {
            "featured_model": "TechMart Phone X",
            "price": "$899",
            "specs": {
                "processor": "Snapdragon 8 Gen 2",
                "memory": "8GB RAM",
                "storage": "256GB (no expansion)",
                "display": '6.7" AMOLED, 120Hz, 2400x1080',
                "camera": "50MP main + 12MP ultrawide + 10MP telephoto",
                "battery": "5000mAh, 65W fast charging",
                "os": "Android 14",
                "connectivity": "5G, Wi-Fi 6E, Bluetooth 5.3",
            },
            "colors": ["Obsidian Black", "Pearl White", "Ocean Blue"],
            "in_stock": True,
            "delivery": "1-2 business days",
        },
        "tablets": {
            "featured_model": "TechMart Tab Pro",
            "price": "$649",
            "specs": {
                "processor": "Apple M2 chip",
                "memory": "8GB RAM",
                "storage": "128GB / 256GB / 512GB options",
                "display": '11" Liquid Retina, 120Hz ProMotion',
                "camera": "12MP rear + 12MP front with Center Stage",
                "battery": "Up to 10 hours",
                "connectivity": "Wi-Fi 6E, optional 5G",
                "accessories": "Compatible with Magic Keyboard and Apple Pencil",
            },
            "colors": ["Silver", "Space Gray"],
            "in_stock": True,
            "delivery": "2-3 business days",
        },
        "audio": {
            "featured_model": "TechMart SoundPro Headphones",
            "price": "$349",
            "specs": {
                "type": "Over-ear wireless",
                "driver": "40mm custom drivers",
                "anc": "Active Noise Cancellation with transparency mode",
                "battery": "30 hours (ANC on), 40 hours (ANC off)",
                "connectivity": "Bluetooth 5.2, 3.5mm jack",
                "features": "Multipoint connection, spatial audio",
                "weight": "250g",
                "charging": "USB-C, 10min charge = 3 hours playback",
            },
            "colors": ["Midnight Black", "Cloud White", "Forest Green"],
            "in_stock": True,
            "delivery": "1-2 business days",
        },
        "accessories": {
            "featured_items": [
                {"name": "USB-C Hub 7-in-1", "price": "$49", "in_stock": True},
                {"name": "Wireless Charging Pad", "price": "$29", "in_stock": True},
                {"name": "Laptop Stand Adjustable", "price": "$39", "in_stock": True},
                {"name": "Screen Protector Kit", "price": "$19", "in_stock": True},
                {"name": "Premium Laptop Bag", "price": "$79", "in_stock": True},
            ],
            "delivery": "1-2 business days",
        },
    }

    product = products.get(product_type.lower())

    if not product:
        return f"Product type '{product_type}' not found. Available categories: {', '.join(products.keys())}"

    if product_type.lower() == "accessories":
        items_list = "\n".join([f"  - {item['name']}: {item['price']}" for item in product["featured_items"]])
        return f"""Accessories Collection:

Featured Items:
{items_list}

Delivery: {product["delivery"]}

All accessories come with a 90-day warranty."""

    specs_list = "\n".join([f"  - {k.title()}: {v}" for k, v in product["specs"].items()])
    colors_list = ", ".join(product.get("colors", []))

    return f"""{product["featured_model"]} - {product["price"]}

Specifications:
{specs_list}

Available Colors: {colors_list}
In Stock: {"Yes" if product["in_stock"] else "No"}
Delivery: {product["delivery"]}

Contact sales for bulk orders or custom configurations."""


@tool
def web_search(keywords: str, region: str = "us-en", max_results: int = 5) -> str:
    """Search the web for updated information.

    Args:
        keywords: Search query keywords
        region: Search region (default: us-en)
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Search results with titles, URLs, and snippets
    """
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)

        if not results:
            return "No search results found for the given keywords."

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   URL: {result.get('href', 'No URL')}\n"
                f"   {result.get('body', 'No description')}"
            )

        return "Web Search Results:\n\n" + "\n\n".join(formatted_results)

    except RatelimitException:
        return "Search rate limit reached. Please try again in a few moments."
    except DuckDuckGoSearchException as e:
        return f"Search error: {e!s}"
    except Exception as e:
        return f"An error occurred during search: {e!s}"


@tool
def get_technical_support(issue_description: str) -> str:
    """Query Bedrock Knowledge Base for troubleshooting and technical assistance.

    Args:
        issue_description: Description of the technical issue or question

    Returns:
        Technical support information and troubleshooting steps from knowledge base
    """
    try:
        # Get KB ID from parameter store
        ssm = boto3.client("ssm")
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        region = boto3.Session().region_name

        kb_id = ssm.get_parameter(Name=f"/{account_id}-{region}/kb/knowledge-base-id")["Parameter"]["Value"]

        kind = ssm.get_parameter(Name="/app/customersupport/kb/type")["Parameter"]["Value"]
        configuration = "managedSearchConfiguration" if kind == "managed" else "vectorSearchConfiguration"
        response = boto3.client("bedrock-agent-runtime", region_name=region).retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": issue_description},
            retrievalConfiguration={configuration: {"numberOfResults": 3}},
        )
        # Managed KB scores are not calibrated to the legacy vector-search cutoff.
        hits = [hit for hit in response.get("retrievalResults", []) if kind == "managed" or hit.get("score", 0) >= 0.4]
        return (
            "\n\n".join(hit.get("content", {}).get("text", "") for hit in hits)
            or "No relevant technical support documentation found."
        )

    except Exception as e:
        raise RuntimeError("Unable to access technical support documentation") from e
