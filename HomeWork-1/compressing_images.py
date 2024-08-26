import os
from PIL import Image

def compress_image(image_path, output_path, quality=85, target_size_kb=100):
    with Image.open(image_path) as img:
        img.save(output_path, optimize=True, quality=quality)
        compressed_size_kb = os.path.getsize(output_path) / 1024
        if compressed_size_kb > target_size_kb:
            # Resize the image to reduce the file size
            img.thumbnail((img.width // 2, img.height // 2))
            img.save(output_path, optimize=True, quality=quality)
            compressed_size_kb = os.path.getsize(output_path) / 1024
            while compressed_size_kb > target_size_kb:
                img.thumbnail((img.width // 2, img.height // 2))
                img.save(output_path, optimize=True, quality=quality)
                compressed_size_kb = os.path.getsize(output_path) / 1024

def compress_images_in_folder(folder_path, size_threshold_kb=100, target_size_kb=100):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                file_size_kb = os.path.getsize(file_path) / 1024
                if file_size_kb > size_threshold_kb:
                    output_path = os.path.join(root, f"{file}")
                    compress_image(file_path, output_path)
                    compressed_size_kb = os.path.getsize(output_path) / 1024
                    print(f"Compressed {file} from {file_size_kb}kb to {compressed_size_kb}kb")
                    if compressed_size_kb > target_size_kb:
                        #print(f"Warning: {file} could not be compressed to below {target_size_kb}kb")
                        pass


# Delete all the created compressed images
def delete_compressed_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')) and file.startswith('compressed_'):
                file_path = os.path.join(root, file)
                os.remove(file_path)


# Example usage
folder_path = 'images'
compress_images_in_folder(folder_path)
# delete_compressed_images(folder_path)
