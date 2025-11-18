import json
import ijson
from decimal import Decimal

INPUT_JSON = "/home/user/ProyectoCienciaDatos/train/data/datos.json"
OUTPUT_JSON = "/home/user/ProyectoCienciaDatos/train/data/datos_reducido.json"

CATEGORY = "Cordialidad"

def convert_decimals(obj):
    """Convierte Decimal → float recursivamente."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(v) for v in obj]
    else:
        return obj


print("🔍 Leyendo JSON gigante en streaming con ijson.items() ...")
result = {}

with open(INPUT_JSON, "rb") as f:
    # Esto devuelve (clave, valor) del nivel superior, uno por uno
    for signer_name, signer_obj in ijson.kvitems(f, ""):

        if CATEGORY not in signer_obj:
            continue

        print(f"✔ Guardando {signer_name}/{CATEGORY}")

        # Filtrar solo Cordialidad
        reduced = {CATEGORY: signer_obj[CATEGORY]}

        # Limpiar Decimals
        reduced = convert_decimals(reduced)

        # Guardar en resultado
        result[signer_name] = reduced


print("💾 Guardando JSON reducido...")
with open(OUTPUT_JSON, "w") as out:
    json.dump(result, out)

print("\n✅ Proceso completado.")
print("📁 Archivo generado:", OUTPUT_JSON)
