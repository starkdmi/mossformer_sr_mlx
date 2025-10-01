// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MossFormer2SR",
    platforms: [
        .iOS(.v16),
        .macOS("13.3"),
    ],
    products: [
        .library(
            name: "MossFormer2SR",
            targets: ["MossFormer2SR"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.20.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.0.0"),
        .package(url: "https://github.com/starkdmi/SwiftAudio", exact: "1.0.0"),
    ],
    targets: [
        .target(
            name: "MossFormer2SR",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
            ],
            path: "Sources/MossFormer2SR"
        ),
        .testTarget(
            name: "MossFormer2SRTests",
            dependencies: [
                "MossFormer2SR",
                .product(name: "AudioUtils", package: "SwiftAudio"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Tests"
        )
    ]
)
