import pyperclip


def get_coordinates():
    points = []
    labels = [
        "Upper",
        "Center",
        "Lower",
        "Inner",
        "Outer",
        "Outer-Upper",
        "Outer-Lower",
        "Inner-Upper",
        "Inner-Lower",
        "Upper-Inner",
        "Upper-Outer",
        "Lower-Center",
    ]

    print("\n=== Left Eye ===")
    for label in labels:
        while True:
            try:
                coord = input(f"Enter {label} point coordinates (x,y): ")
                x, y = map(float, coord.split(","))
                if 0 <= x <= 1024 and 0 <= y <= 600:
                    points.append([round(x / 1024, 3), round(y / 600, 3)])
                    break
                print("Coordinates out of bounds (0-1024, 0-600)")
            except ValueError:
                print("Invalid format. Use x,y (e.g., 100.5,200.3)")

    print("\n=== Right Eye ===")
    for label in labels:
        while True:
            try:
                coord = input(f"Enter {label} point coordinates (x,y): ")
                x, y = map(float, coord.split(","))
                if 0 <= x <= 1024 and 0 <= y <= 600:
                    points.append([round(x / 1024, 3), round(y / 600, 3)])
                    break
                print("Coordinates out of bounds (0-1024, 0-600)")
            except ValueError:
                print("Invalid format. Use x,y (e.g., 100.5,200.3)")

    return points


def create_class_string(points):
    class_str = """class NewExpression(BaseExpression):
    def define_keyframes(self):
        return [
            {}
        ], {}"""
    formatted_points = str(points).replace("], [", "],\n                [")
    return class_str.format(formatted_points, "{}")


def main():
    print("Enter coordinates for 24 points (12 per eye)")
    points = get_coordinates()
    class_string = create_class_string(points)
    pyperclip.copy(class_string)
    print("\nClass has been copied to clipboard!")
    print("Preview of the class:")
    print(class_string)


if __name__ == "__main__":
    main()
