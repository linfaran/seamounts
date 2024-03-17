from PIL import Image
def convert_32bit_to_24bit_png(input_path, output_path):
    # Open the 32-bit PNG image
    image_32bit = Image.open(input_path)

    # Check if the image has an alpha channel
    if image_32bit.mode == 'RGBA':
        # Convert the image to 'RGB' mode which is 24-bit
        image_24bit = image_32bit.convert('RGB')
        # Save the converted image
        image_24bit.save(output_path, format='PNG')
    else:
        print("The image is not in RGBA mode.")

# Path to the 32-bit PNG image
input_image_path = 'D:/software/Python/PythonProject/Pytorch-UNet-master/data/test/1.png'
# Path to save the 24-bit PNG image
output_image_path = 'D:/software/Python/PythonProject/Pytorch-UNet-master/data/test/24bit.png'

convert_32bit_to_24bit_png(input_image_path, output_image_path)
