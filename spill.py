import os

img_dir = r"D:\final_system\line_seg\archive_12\Documents\Documents\Receipt\img"
json_dir = r"D:\final_system\line_seg\archive_12\Documents\Documents\Receipt\ann"


def add_suffix(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if not os.path.isfile(path):
            continue

        name, ext = os.path.splitext(file)

        new_name = name + "RE" + ext
        new_path = os.path.join(folder, new_name)

        os.rename(path, new_path)


# إضافة AF للصور
add_suffix(img_dir)

# إضافة AF لملفات json
add_suffix(json_dir)

print("تم إضافة AF لكل الملفات بنجاح")