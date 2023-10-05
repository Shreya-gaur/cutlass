// Limitations of the kernel --> 1. Only takes in multiples of 32 for C
// 								 2. Cannot handle big numbers of K
// Notes : Depth Size change in filter --> 1. P4ConvZ2 -- K * C * output_stablizer 


#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"


// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = float;  // <- data type of epilogue operations
using ElementInputA = float;              // <- data type of elements in input matrix A
using ElementInputB = float;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassSimt;

using SmArch = cutlass::arch::Sm60;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;  // Threadblock tile shape

using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;         // Warp tile shape

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;    // TensorCore instruction shape

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    1,                                                 // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

constexpr int NumStages = 2;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord rotated_filter_size;
  cutlass::Tensor4DCoord output_ro;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  bool rotate_filter;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(false),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    rotate_filter(false),
    alpha(1),
    beta(0),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of int4b_t elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 32 elements.
    //
    int const kAlignment = 32;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor4DCoord input_size,
    cutlass::Tensor4DCoord filter_size) {

	std::cout<<"Update is called here"<<std::endl;

    this->input_size = input_size;
    this->filter_size = filter_size;

	if(rotate_filter){
		this->rotated_filter_size.h() = this->filter_size.h();
		this->rotated_filter_size.w() = this->filter_size.w();
	}

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("rotate-filter")) {
      rotate_filter = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c(); 
	

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.h() == 3 && filter_size.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      padding.n() = filter_size.h() / 2;
   	  padding.h() = filter_size.h() / 2;
   	  padding.w() = filter_size.w() / 2;
   	  padding.c() = filter_size.w() / 2;
     // filter_size.h() = 1;
     // filter_size.w() = 1;
     // padding = {0, 0, 0, 0};
    }

	if (rotate_filter){
	  rotated_filter_size.n() = filter_size.n() * 2;
	  rotated_filter_size.h() = filter_size.h();
	  rotated_filter_size.w() = filter_size.w();
	  rotated_filter_size.c() = filter_size.c();
	  
	  output_ro.n() = input_size.n();
	  output_ro.h() = (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1;
      output_ro.w() = (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1;
	  output_ro.c() = filter_size.n() * 2; 

	  std::cout<<"Rotated Output Dimensions"<< "\n";
	  
	  std::cout<<"output_ro.n()="<< output_ro.n() << "\n";
	  std::cout<<"output_ro.h()="<< output_ro.h() << "\n";
	  std::cout<<"output_ro.w()="<< output_ro.w() << "\n";
	  std::cout<<"output_ro.c()="<< output_ro.c() << "\n";
	}
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "52_volta_simt_conv2dfprop example\n\n"
      << "  This example uses Turing's Tensor Core operators on int4 data types to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n=<int>            Input tensor extent N\n"
      << "  --h=<int>            Input tensor extent H\n"
      << "  --w=<int>            Input tensor extent W\n"
      << "  --c=<int>            Input tensor extent C\n"
      << "  --k=<int>            Filter extent K\n"
      << "  --r=<int>            Filter extent R\n"
      << "  --s=<int>            Filter extent S\n\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
	  << "  --rotate-filter		 If set, it rotates the filter and computes convolution. \n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/52_volta_simt_conv2dfprop/52_volta_simt_conv2dfprop/ --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/52_volta_simt_conv2dfprop/52_volta_simt_conv2dfprop/  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

    return out;
  }
  
  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
	return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};
struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result(): 
    runtime_ms(0), 
    gflops(0),
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,P,Q,H,W,C,K,R,S,Runtime,GFLOPs";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out 
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << (options.input_size.h() + options.padding.n() + options.padding.h() - options.filter_size.h()) / options.conv_stride.row() + 1 << ","
      << (options.input_size.w() + options.padding.w() + options.padding.c() - options.filter_size.w()) / options.conv_stride.column() + 1 << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

/// Runs one benchmark
Result profile_convolution(Options const &options) {

  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //
	  std::cout<<"Output Dimensions"<< "\n";
	  
	  std::cout<<"output_size="<< options.output_size() << "\n";

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b_rotated(options.rotated_filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_ro(options.output_ro);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_c(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_c_ro(options.output_ro);

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);

  // Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8),
      0);

  // Fill tensor C on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_b_rotated.host_view());
  
  // Fill tensor C on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_c.host_view());
  
  cutlass::reference::host::TensorFill(
      tensor_c_ro.host_view());

  // Fill tensor C for reference on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_c.host_view());

  cutlass::reference::host::TensorFill(
      tensor_ref_c_ro.host_view());
  
  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_ref_c.sync_device();

  //
  // Define arguments for CUTLASS Convolution
  //

  // mode (kCrossCorrelation or kConvolution)
  cutlass::conv::Mode mode = cutlass::conv::Mode::kConvolution;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Construct Conv2dProblemSize with user defined output size
  cutlass::conv::Conv2dProblemSize problem_size(      
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      split_k_slices);


  // Construct ImplicitGemm::Argument structure with conv2d 
  // problem size, data pointers, and epilogue values
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c.device_ref(),
    tensor_c.device_ref(),
    {options.alpha, options.beta},
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm implicit_gemm_op;

  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(result.status);

  result.status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  //
  // Launch initialized CUTLASS kernel
  //
  result.status = implicit_gemm_op();

  CUTLASS_CHECK(result.status);

  //
  // Optional reference check
  //
  
  if (options.reference_check) {
    std::cout << "Verification on host...\n";

    // Compute with reference implementation
	if(options.rotate_filter){
			
   	 cutlass::reference::host::Conv2dFpropEq<
   	   ElementInputA,
   	   LayoutInputA,
   	   ElementInputB,
   	   LayoutInputB,
   	   ElementOutput,
   	   LayoutOutput,
   	   ElementComputeEpilogue,
   	   ElementAccumulator,
   	   ElementOutput,
   	   cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>
   	 >(
   	   problem_size,
   	   tensor_a.host_ref(),
   	   tensor_b.host_ref(),
  	   tensor_b_rotated.host_ref(),
   	   tensor_c_ro.host_ref(),
   	   tensor_ref_c_ro.host_ref(),
   	   options.alpha,
   	   options.beta
   	  );
	 
	}

	else{

   		 cutlass::reference::host::Conv2dFprop<
   		   ElementInputA,
   		   LayoutInputA,
   		   ElementInputB,
   		   LayoutInputB,
   		   ElementOutput,
   		   LayoutOutput,
   		   ElementComputeEpilogue,
   		   ElementAccumulator,
   		   ElementOutput,
   		   cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>
   		 >(
   		   problem_size,
   		   tensor_a.host_ref(),
   		   tensor_b.host_ref(),
   		   tensor_c.host_ref(),
   		   tensor_ref_c.host_ref(),
   		   options.alpha,
   		   options.beta
	    );

	}

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_c.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(
      tensor_c.host_view(),
      tensor_ref_c.host_view());

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    }
    else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  }
  else {
    result.reference_check = cutlass::Status::kInvalid;
  }

  if (options.save_workspace) {

    std::stringstream ss;

    ss << "52_volta_simt_workspace_conv2dfprop_"
      << options.input_size.n() << "x" << options.input_size.h() << "x" << options.input_size.w() << "x" << options.input_size.c() 
      << "_"
      << options.filter_size.n() << "x" << options.filter_size.h() << "x" << options.filter_size.w() << "x" << options.filter_size.c() 
      << ".dat";

    std::ofstream output_workspace(ss.str());
	if (options.rotate_filter) {
    	output_workspace 
      		<< "Input = \n" << tensor_a.host_view() << "\n\n"
      		<< "Filters = \n" << tensor_b.host_view() << "\n\n"
	  		<< "Rotated Filters = \n" << tensor_b_rotated.host_view() << "\n\n";
	}
	else{
   		output_workspace 
   	   		<< "Input = \n" << tensor_a.host_view() << "\n\n"
   	   		<< "Filters = \n" << tensor_b.host_view() << "\n\n";
	}

	if (options.reference_check) {
		if (options.rotate_filter) {
   		   output_workspace << "Reference = \n" << tensor_ref_c_ro.host_view() << "\n\n";
   		}
		else{
     	   output_workspace << "Reference = \n" << tensor_ref_c.host_view() << "\n\n";
		}
    }

    output_workspace << "Computed = \n" << tensor_c.host_view() << std::endl;

    std::cout << "Results written to '" << ss.str() << "'." << std::endl;
  }
  
  //
  // Performance measurement
  //

  if (options.measure_performance) {

    cudaEvent_t events[2];
    
    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }
    }

    // Record an event at the start of a series of convolution operations.
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Launch a sequence of implicit GEMM operations on the device
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      result.status = implicit_gemm_op();
      CUTLASS_CHECK(result.status);
    }

    // Record an event when the convolutions have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Print average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }
  }

  return result;
}


int main(int argc,char const **args) {

  // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
  //
  // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
    std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

    // Returning zero when built on older Toolkits so tests pass. The actions of this SDK example are no-op.
    return 0;
  }
  else {
    
  	Options options;
  
  	options.parse(argc, args);

  	if (options.help) {
   		 options.print_usage(std::cout) << std::endl;
   		return 0;
  	}

    if (!options.valid()) {
      std::cerr << "Invalid problem." << std::endl;
      return -1;
    }

    Result result = profile_convolution(options);

    Result::print_header(std::cout, options) << std::endl;
    result.print(std::cout, 1, options) << std::endl;

	return 0;
  }
}
