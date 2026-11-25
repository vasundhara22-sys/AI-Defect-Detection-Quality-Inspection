import json
from pathlib import Path

REGISTRY = Path(__file__).with_name("products.json")


def load_products():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def get_product(product_id):
    products = load_products()
    if product_id not in products:
        raise KeyError(f"Unknown product '{product_id}'. Known: {', '.join(products)}")
    return products[product_id]


if __name__ == "__main__":
    for pid, p in load_products().items():
        print(f"{pid:15} model={p['model']}  fail>={p['fail_conf']}  review>={p['review_conf']}")
        print(f"{'':15} defects={', '.join(p['defects'])}")
        print(f"{'':15} status={p['status']}")
