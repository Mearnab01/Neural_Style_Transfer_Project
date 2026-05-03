import os

folders = [
    "templates/partials",
]

files = [
    "templates/base.html",
    "templates/index.html",
    "templates/partials/_topbar.html",
    "templates/partials/_sidebar.html",
    "templates/partials/_canvas.html",
    "templates/partials/_examples.html",
    "templates/partials/_faq.html",
    "templates/partials/_footer.html",
    "templates/partials/_loader.html",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

for file in files:
    with open(file, "w") as f:
        pass
    print(f"Created file: {file}")

print("\nDone.")