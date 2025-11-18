import json
import ijson
from decimal import Decimal

INPUT_JSON = r"C:\Users\Zabdiel Julian\Downloads\Proyectos\ProyectoCienciaDatos\train\dataset\datos.json"
OUTPUT_JSON = r"C:\Users\Zabdiel Julian\Downloads\Proyectos\ProyectoCienciaDatos\train\dataset\datosReducidos.json"

CATEGORY = "Cordialidad"

def convert_decimals(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(v) for v in obj]
    else:
        return obj

result = {}

print("Detectando estructura del JSON...")

with open(INPUT_JSON, "rb") as f:
    parser = ijson.parse(f)
    for prefix, event, value in parser:
        if event == "map_key":
            json_type = "object"
            break
        elif event == "start_array":
            json_type = "array"
            break

print("Estructura detectada:", json_type)

print("Leyendo JSON gigante...")

with open(INPUT_JSON, "rb") as f:

    if json_type == "object":
        # Recorrer los signers del objeto raíz
        for signer_name, signer_obj in ijson.kvitems(f, ""):

            if CATEGORY not in signer_obj:
                continue

            print(f"Guardando {signer_name}/{CATEGORY}")

            reduced = {CATEGORY: signer_obj[CATEGORY]}
            reduced = convert_decimals(reduced)
            result[signer_name] = reduced

    else:
        # Recorrer una lista de signers
        for signer_obj in ijson.items(f, "item"):

            signer_name = signer_obj.get("signer") or signer_obj.get("id")

            if not signer_name:
                continue

            if CATEGORY not in signer_obj:
                continue

            print(f"Guardando {signer_name}/{CATEGORY}")

            reduced = {CATEGORY: signer_obj[CATEGORY]}
            reduced = convert_decimals(reduced)
            result[signer_name] = reduced

print("Guardando JSON reducido...")
with open(OUTPUT_JSON, "w") as out:
    json.dump(result, out)

print("\nProceso completado.")
print("Archivo generado:", OUTPUT_JSON)
