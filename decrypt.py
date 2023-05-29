from pathlib import Path
import re
import subprocess

trace = Path("dbtrace.log").read_text()
results = re.findall(r"sqlcipher '(.*msg_.\.db)'\n.*\n(.*\n.*\n.*\n.*)", trace)
results = sorted(set(results))

for db_path, sql in results:
    db_path = Path(db_path)
    plain_path = Path(f"plain_{db_path.name}")
    plain_path.unlink(missing_ok=True)

    print(f"Decrypting {db_path} into {plain_path} ...")

    cmd = ["sqlcipher", str(db_path)]
    sql += f"""ATTACH DATABASE '{plain_path}' AS plaintext KEY '';
SELECT sqlcipher_export ('plaintext');
DETACH DATABASE plaintext;"""

    p = subprocess.run(
        cmd, input=sql.encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(p.stdout.strip().decode())
