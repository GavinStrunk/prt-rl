import pygame

def main():
    pygame.init()
    pygame.joystick.init()

    # Check for available controllers
    if pygame.joystick.get_count() == 0:
        print("No controller found.")
        return

    # Initialize the controller
    controller = pygame.joystick.Joystick(0)
    controller.init()

    # Get controller name
    print("Controller name:", controller.get_name())

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.JOYBUTTONDOWN:
                print("Button pressed:", event.button)

            if event.type == pygame.JOYHATMOTION:
                print("D-pad up pressed ", event.value)

            if event.type == pygame.JOYAXISMOTION:
                print("Axis moved:", event.axis, event.value)

    pygame.quit()

if __name__ == "__main__":
    main()