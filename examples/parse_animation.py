import pyperclip


def get_coordinates(use_offset=False):
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
    left_eye_points = []
    for label in labels:
        while True:
            try:
                coord = input(f"Enter {label} point coordinates (x,y): ")
                x, y = map(float, coord.split(","))
                if 0 <= x <= 1024 and 0 <= y <= 600:
                    left_eye_points.append([x, y])  # Store raw coordinates
                    points.append([round(x / 1024, 3), round(y / 600, 3)])
                    break
                print("Coordinates out of bounds (0-1024, 0-600)")
            except ValueError:
                print("Invalid format. Use x,y (e.g., 100.5,200.3)")

    print("\n=== Right Eye ===")
    if use_offset:
        while True:
            try:
                x_offset = float(input("Enter x-axis offset for right eye: "))
                if -1024 <= x_offset <= 1024:
                    # Apply offset to left eye coordinates and normalize
                    for i, [x, y] in enumerate(left_eye_points):
                        new_x = x + x_offset
                        if 0 <= new_x <= 1024:
                            points.append(
                                [round(new_x / 1024, 3), round(y / 600, 3)]
                            )
                        else:
                            print("Offset would put coordinates out of bounds")
                            return get_coordinates(use_offset)
                    break
                print("Offset out of bounds (-1024 to 1024)")
            except ValueError:
                print("Invalid format. Enter a number (e.g., 200)")
    else:
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
    keyframes = [
            {}
        ]"""
    formatted_points = str(points).replace("], [", "],\n            [")
    return class_str.format(formatted_points)


def main():
    print("Enter coordinates for 24 points (12 per eye)")
    use_offset = (
        input("Use x-axis offset for right eye? (y/n): ").lower() == "y"
    )
    points = get_coordinates(use_offset)
    class_string = create_class_string(points)
    pyperclip.copy(class_string)
    print("\nClass has been copied to clipboard!")
    print("Preview of the class:")
    print(class_string)


if __name__ == "__main__":
    main()
