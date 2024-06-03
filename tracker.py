import math

class ObjectTracker:
    def __init__(self):
        # Dictionary to store the center points of objects with their corresponding IDs
        self.object_centers = {}
        self.next_id = 0

    def update(self, bounding_boxes):
        updated_objects = []
        
        # Iterate through each bounding box
        for bbox in bounding_boxes:
            x, y, width, height = bbox
            center_x = (x + x + width) // 2
            center_y = (y + y + height) // 2
            
            detected_existing_object = False
            
            for obj_id, center in self.object_centers.items():
                distance = math.hypot(center_x - center[0], center_y - center[1])
                
                # If the distance is less than 35, consider it the same object
                if distance < 35:
                    self.object_centers[obj_id] = (center_x, center_y)
                    updated_objects.append([x, y, width, height, obj_id])
                    detected_existing_object = True
                    break
            
            if not detected_existing_object:
                self.object_centers[self.next_id] = (center_x, center_y)
                # Add the bounding box and its new ID to the list of updated objects
                updated_objects.append([x, y, width, height, self.next_id])
                self.next_id += 1

        # Create a new dictionary to keep only the center points of the objects detected in the current frame
        new_object_centers = {}
        for obj in updated_objects:
            _, _, _, _, obj_id = obj
            new_object_centers[obj_id] = self.object_centers[obj_id]

        # Update the stored object centers to only include those from the current frame
        self.object_centers = new_object_centers.copy()
        
        return updated_objects
