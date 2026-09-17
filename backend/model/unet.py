import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


class UNet(nn.Module):

    """
    2D U-Net for HVSMR cardiac structure segmentation.

    Input:
        1-channel MRI slice

    Output:
        9 segmentation classes

    Labels:
        0 = Background
        1 = Left Ventricle
        2 = Right Ventricle
        3 = Left Atrium
        4 = Right Atrium
        5 = Aorta
        6 = Pulmonary Artery
        7 = SVC
        8 = IVC
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=9
    ):

        super().__init__()

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.down1 = DoubleConv(
            in_channels,
            64
        )

        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(
            64,
            128
        )

        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(
            128,
            256
        )

        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(
            256,
            512
        )

        self.pool4 = nn.MaxPool2d(2)

        # -------------------------------------------------
        # Bridge
        # -------------------------------------------------

        self.bridge = DoubleConv(
            512,
            1024
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.conv4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.conv3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.conv2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv1 = DoubleConv(
            128,
            64
        )

        # -------------------------------------------------
        # Final segmentation layer
        # -------------------------------------------------

        self.final = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    # =====================================================
    # Helper: Match Spatial Dimensions
    # =====================================================

    @staticmethod
    def match_size(x, reference):

        """
        Make x have the same H,W as reference.

        This protects the skip connections from
        dimension mismatches caused by pooling.
        """

        diff_y = reference.size(2) - x.size(2)
        diff_x = reference.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:

            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2
                ]
            )

        return x

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):

        original_height = x.size(2)
        original_width = x.size(3)

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        d1 = self.down1(x)

        p1 = self.pool1(d1)

        d2 = self.down2(p1)

        p2 = self.pool2(d2)

        d3 = self.down3(p2)

        p3 = self.pool3(d3)

        d4 = self.down4(p3)

        p4 = self.pool4(d4)

        # -------------------------------------------------
        # Bridge
        # -------------------------------------------------

        bridge = self.bridge(p4)

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        u4 = self.up4(bridge)

        u4 = self.match_size(
            u4,
            d4
        )

        u4 = torch.cat(
            [u4, d4],
            dim=1
        )

        u4 = self.conv4(u4)

        # -------------------------------------------------

        u3 = self.up3(u4)

        u3 = self.match_size(
            u3,
            d3
        )

        u3 = torch.cat(
            [u3, d3],
            dim=1
        )

        u3 = self.conv3(u3)

        # -------------------------------------------------

        u2 = self.up2(u3)

        u2 = self.match_size(
            u2,
            d2
        )

        u2 = torch.cat(
            [u2, d2],
            dim=1
        )

        u2 = self.conv2(u2)

        # -------------------------------------------------

        u1 = self.up1(u2)

        u1 = self.match_size(
            u1,
            d1
        )

        u1 = torch.cat(
            [u1, d1],
            dim=1
        )

        u1 = self.conv1(u1)

        # -------------------------------------------------
        # Final prediction
        # -------------------------------------------------

        output = self.final(u1)

        # -------------------------------------------------
        # Restore original image size
        # -------------------------------------------------

        output = output[
            :,
            :,
            :original_height,
            :original_width
        ]

        return output