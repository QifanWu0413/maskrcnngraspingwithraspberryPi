def load_mask(self, image_id):
    """Generate instance masks for an image.
   Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a balloon dataset image, delegate to parent class.
    image_info = self.image_info[image_id]
    if image_info["source"] != "balloon" :
        return super(self.__class__, self).load_mask(image_id)

    name_id = image_info["class_id"]
    # Convert polygons to a bitmap mask of shape
    # [height, width, instance_count]
    info = self.image_info[image_id]
    mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)
    class_ids = np.array(name_id, dtype=np.int32)

    for i, p in enumerate(info["polygons"]):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        mask[rr, cc, i] = 1
    # print( mask.astype(np.bool), name_id)

    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID only, we return an array of 1s
    return (mask.astype(np.bool), class_ids)
