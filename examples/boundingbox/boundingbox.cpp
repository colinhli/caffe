#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */

class Regression
{
public:
	Regression(const string& model_file,
			   const string& trained_file,
			   const string& mean_file,
			   const string& mean_value, int gpu = -1); 
	std::vector<float> regression(const cv::Mat& img); 
private:
  void SetMean(const string& mean_file, const string& mean_value); 
  std::vector<float> Predict(const cv::Mat& img); 
  void WrapInputLayer(std::vector<cv::Mat>* input_channels); 
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels); 
private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Regression::Regression(const string& model_file,
							   const string& trained_file,
							   const string& mean_file,
							   const std::string& mean_value, int gpu /*= -1*/)
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	if (gpu < 0)
	{
		Caffe::set_mode(Caffe::CPU);
	}
	else
	{
		Caffe::SetDevice(gpu);
		Caffe::set_mode(Caffe::GPU);
	} 
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
			<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height()); 
	/* Load the binaryproto mean file. */
	SetMean(mean_file, mean_value); 
}

static bool PairCompare(const std::pair<float, int>& lhs,
						const std::pair<float, int>& rhs)
{
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N)
{
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<float> Regression::regression(const cv::Mat& img)
{
	std::vector<float> output = Predict(img);
	// LOG(INFO) << output.size(); 
  // LOG(INFO) << output[0] << " " << output[1] << " " << output[2] << " " << output[3];
	std::vector<float> box;

	float tx = output[0];
	float ty = output[1];
	float tw = output[2];
	float th = output[3];

	float w = exp(tw) * input_geometry_.width;
	float h = exp(th) * input_geometry_.height;
	float x_offset = tx * input_geometry_.width;
	float y_offset = ty * input_geometry_.height;

	float x = x_offset + input_geometry_.width / 2 - w / 2;
	float y = y_offset + input_geometry_.height / 2 - h / 2;

	box.push_back(x);
	box.push_back(y);
	box.push_back(w);
	box.push_back(h);

	return box;
}

/* Load the mean file in binaryproto format. */
void Regression::SetMean(const string& mean_file, const string& mean_value)
{
	cv::Scalar channel_mean;
	if (!mean_file.empty())
	{
		CHECK(mean_value.empty()) << "Cannot specify mean_file and mean_value at the same time";
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

		/* Convert from BlobProto to Blob<float> */
		Blob<float> mean_blob;
		mean_blob.FromProto(blob_proto);
		CHECK_EQ(mean_blob.channels(), num_channels_)
				<< "Number of channels of mean file doesn't match input layer.";

		/* The format of the mean file is planar 32-bit float BGR or grayscale. */
		std::vector<cv::Mat> channels;
		float* data = mean_blob.mutable_cpu_data();
		for (int i = 0; i < num_channels_; ++i)
		{
			/* Extract an individual channel. */
			cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
			channels.push_back(channel);
			data += mean_blob.height() * mean_blob.width();
		}

		/* Merge the separate channels into a single image. */
		cv::Mat mean;
		cv::merge(channels, mean);

		/* Compute the global mean pixel value and create a mean image
		 * filled with this value. */
		// channel_mean = cv::mean(mean);
		// mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
		mean_ = mean;
	}
	if (!mean_value.empty())
	{
		CHECK(mean_file.empty()) <<
								 "Cannot specify mean_file and mean_value at the same time";
		stringstream ss(mean_value);
		vector<float> values;
		string item;
		while (getline(ss, item, ','))
		{
			float value = std::atof(item.c_str());
			values.push_back(value);
		}
		CHECK(values.size() == 1 || values.size() == num_channels_) <<
				"Specify either 1 mean_value or as many as channels: " << num_channels_;

		std::vector<cv::Mat> channels;
		for (int i = 0; i < num_channels_; ++i)
		{
			/* Extract an individual channel. */
			cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
							cv::Scalar(values[i]));
			channels.push_back(channel);
		}
		cv::merge(channels, mean_);
	}
}

std::vector<float> Regression::Predict(const cv::Mat& img)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	LOG(INFO) << input_geometry_.width << " " << input_geometry_.height;
	input_layer->Reshape(1, num_channels_,
						 input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];  
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();

	// int count = output_layer->num();
  // const float* begin = output_layer->cpu_data(); 
  // const float* end = begin + count; 
  LOG(INFO) << begin[0] << " " << begin[1] << " " << begin[2] << " " << begin[3];
  return std::vector<float>(begin, end); 
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Regression::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Regression::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
  if (mean_.data == nullptr)
  {
      sample_normalized = sample_float;
  }
	else
  {
      cv::subtract(sample_float, mean_, sample_normalized);
  }
	
	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		  == net_->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv)
{
	if (argc != 4)
	{
		std::cerr << "Usage: " << argv[0]
				  << " deploy.prototxt network.caffemodel"
				  << " img.jpg" << std::endl;
		return 1;
	}
	FLAGS_alsologtostderr = 1;
	::google::InitGoogleLogging(argv[0]);

	string model_file   = argv[1];
	string trained_file = argv[2];

	Regression regression(model_file, trained_file, "", "");

	string file = argv[3];

	std::cout << "---------- Prediction for "
			  << file << " ----------" << std::endl;

	cv::Mat img = cv::imread(file, -1);
	CHECK(!img.empty()) << "Unable to decode image " << file;

	cv::Mat input_image;
	cv::resize(img, input_image, cv::Size(200, 40));
	float scale_x = (float)(input_image.cols) / img.cols;
	float scale_y = (float)(input_image.rows) / img.rows;
	std::vector<float> box = regression.regression(img);
	LOG(INFO) << box.size();
	// std::vector<Prediction> predictions = classifier.Classify(img);

	// /* Print the top N predictions. */
	for (size_t i = 0; i < box.size(); ++i)
	{
		std::cout << std::fixed << std::setprecision(4) << box[i] << std::endl;
	}
	cv::Rect rect;
	rect.x = box[0] / scale_x;
	rect.y = box[1] / scale_y;
	rect.width = box[2] / scale_x;
	rect.height = box[3] / scale_y;

	cv::rectangle(img, rect.tl(), rect.br(), cv::Scalar(0, 0, 255), 1);
	cv::imshow("demo", img);
	cv::waitKey(-1);
}
#else
int main(int argc, char** argv)
{
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
