#include <iostream>
#include <cstring>

class CDataBlobKernel
{
public:
	int rows;
	int cols;
	int channels;	 // in element
	int channelStep; // in byte
	float *data;

private:
	int *p_rows = &rows;
	int *p_cols = &cols;
	int *p_channels = &channels;
	int *p_channelStep = &channelStep;
	float **p_data = &data;

public:
	CDataBlobKernel()
	{
		rows = 0;
		cols = 0;
		channels = 0;
		channelStep = 0;
		data = nullptr;
	}

	CDataBlobKernel(int _rows, int _cols, int _channels, int _channelStep, float *_data)
	{
		this->rows = _rows;
		this->cols = _cols;
		this->channels = _channels;
		this->channelStep = _channelStep;

		size_t size_bytes = size_t(rows) * cols * channelStep;

		std::memcpy(this->data, _data, size_bytes);

		if (data == nullptr)
		{
			std::cerr << "CDataBlobKernel Failed to alloc memeory for uint8 data blob: "
					  << _rows << "*"
					  << _cols << "*"
					  << _channels << std::endl;
		}
	}

	~CDataBlobKernel()
	{
	}
};

class FiltersKernel
{

public:
	int channels;
	int num_filters;
	bool is_depthwise;
	bool is_pointwise;
	bool with_relu;
	CDataBlobKernel weights;
	CDataBlobKernel biases;

private:
	FiltersKernel()
	{
		channels = 0;
		num_filters = 0;
		is_depthwise = false;
		is_pointwise = false;
		with_relu = true;
	}
	FiltersKernel(int channels,
				  int num_filters,
				  bool is_depthwise,
				  bool is_pointwise,
				  bool with_relu,

				  int weights_rows,
				  int weights_cols,
				  int weights_channels,
				  int weights_channelStep,
				  float *weight_data,

				  int biases_rows,
				  int biases_cols,
				  int biases_channels,
				  int biases_channelStep,
				  float *biases_data)
	{
		this->channels = channels;
		this->num_filters = num_filters;
		this->is_depthwise = is_depthwise;
		this->is_pointwise = is_pointwise;
		this->with_relu = with_relu;

		weights = CDataBlobKernel(weights_rows, weights_cols, weights_channels, weights_channelStep, weight_data);
		biases = CDataBlobKernel(biases_rows, biases_cols, biases_channels, biases_channelStep, biases_data);
	}
};

bool convolution_1x1pointwiseKernel(int input_rows, 
									int input_cols, 
									int input_channels, 
									int input_channelStep, 
									float *input_data,


									int channels,
									int num_filters,
									bool is_depthwise,
									bool is_pointwise,
									bool with_relu,

									int weights_rows,
									int weights_cols,
									int weights_channels,
									int weights_channelStep,
									float *weight_data,

									int biases_rows,
									int biases_cols,
									int biases_channels,
									int biases_channelStep,
									float *biases_data, 
									
									
									int output_rows, 
									int output_cols, 
									int output_channels, 
									int output_channelStep, 
									float *output_data);