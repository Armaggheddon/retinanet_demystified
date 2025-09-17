import torch
import torchvision.models as models

def calculate_resnet_strides(backbone_name, input_size=(224, 224)):
    print(f"Calculating strides for backbone: {backbone_name}")
    resnet = models.get_model(backbone_name, weights=None)

    dummy_input = torch.randn(1, 3, *input_size)

    # Pass the input through the layers to get C3, C4, C5 feature maps
    x = resnet.conv1(dummy_input)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x) # Output of initial downsampling (stride 4)

    x = resnet.layer1(x) # C2 feature map
    c3 = resnet.layer2(x) # C3 feature map
    c4 = resnet.layer3(c3) # C4 feature map
    c5 = resnet.layer4(c4) # C5 feature map

    # Calculate the strides by dividing the input size by the feature map sizes
    c3_stride = input_size[0] // c3.shape[2]
    c4_stride = input_size[0] // c4.shape[2]
    c5_stride = input_size[0] // c5.shape[2]

    # FPN adds P6 and P7, each with stride of 2 
    p6_stride = c5_stride * 2
    p7_stride = p6_stride * 2

    fpn_strides = (c3_stride, c4_stride, c5_stride, p6_stride, p7_stride)

    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  C3 Shape: {c3.shape} -> Stride: {c3_stride}")
    print(f"  C4 Shape: {c4.shape} -> Stride: {c4_stride}")
    print(f"  C5 Shape: {c5.shape} -> Stride: {c5_stride}")
    print(f"  Final FPN Strides (P3-P7): {fpn_strides}\n")

    return fpn_strides


if __name__ == "__main__":
    supported_backends = [
        "resnet18", 
        "resnet34", 
        "resnet50", 
        "resnet101", 
        "resnet152"
    ]
    
    all_strides = {}
    for backbone in supported_backends:
        strides = calculate_resnet_strides(backbone, input_size=(224, 224))
        if strides:
            all_strides[backbone] = strides

    # Verify that all calculated strides are the same
    first_stride_set = next(iter(all_strides.values()))
    if all(s == first_stride_set for s in all_strides.values()):
        print("="*40)
        print("Conclusion: All tested ResNet backbones have the same FPN strides.")
        print(f"Strides (P3-P7): {first_stride_set}")
        print("="*40)