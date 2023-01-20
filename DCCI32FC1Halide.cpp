#include <Halide.h>

using namespace Halide;

Halide::Expr calcWeight(Halide::Expr grad)
{
	return cast<float>(Expr(1.0) / (Expr(1.0) + pow(cast<double>(grad), 5)));
}

Halide::Func calcL1Norm(Halide::Func base, std::pair<Halide::Expr, Halide::Expr> offsets)
{
	vector<Var> args = base.args();
	Func norm;
	norm(args[0], args[1]) = cast<float>(abs(base(args[0], args[1]) - base(args[0] + offsets.first, args[1] + offsets.second)));
	return norm;
}

Halide::Expr calcGradUpRightGray(Halide::Func base)
{
	vector<Var> args = base.args();
	Func norm = calcL1Norm(base, { -2, 2 });
	Expr grad = 0.f;

	int mn[3] = { 3, 1, -1 };
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			grad += norm(args[0] + mn[i], args[1] - mn[j]);
		}
	}
	return grad;
}

Halide::Expr calcGradDownRightGray(Halide::Func base)
{
	vector<Var> args = base.args();
	Func norm = calcL1Norm(base, { -2, -2 });
	Expr grad = 0.f;

	int mn[3] = { 3, 1, -1 };
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			grad += norm(args[0] + mn[i], args[1] + mn[j]);
		}
	}
	return grad;
}

Halide::Expr calcGradHorizontalGray(Halide::Func base)
{
	vector<Var> args = base.args();
	Func norm = calcL1Norm(base, { 2, 0 });
	Expr grad = 0.f;

	int m[2] = { 0, 2 };
	int n[2] = { -1, 1 };
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			grad += norm(args[0] - m[i], args[1] + n[j]);
		}
	}
	int n2[3] = { -2, 0, 2 };
	for (int i = 0; i < 3; i++)
	{
		grad += norm(args[0] - 1, args[1] + n2[i]);
	}

	return grad;
}

Halide::Expr calcGradVerticalGray(Halide::Func base)
{
	vector<Var> args = base.args();
	Func norm = calcL1Norm(base, { 0, 2 });
	Expr grad = 0.f;

	int m[2] = { -1, 1 };
	int n[2] = { 0, 2 };
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			grad += norm(args[0] + m[i], args[1] - n[j]);
		}
	}
	int m2[3] = { -2, 0, 2 };
	for (int i = 0; i < 3; i++)
	{
		grad += norm(args[0] + m2[i], args[1] - 1);
	}

	return grad;
}

/// <summary>
/// float in float out, 0.f~1.fを前提とする
/// </summary>
// 146(end) - 11(start) - 0(blanc comment) + 1 + 11(called function) = 147
class DCCI32FC1Halide : public Halide::Generator<DCCI32FC1Halide>
{
public:
	GeneratorInput<Buffer<float>> input{ "input", 2 };
	GeneratorOutput<Buffer<float>> output{ "output", 2 };
	GeneratorInput<float> threshold{ "threshold", 1.15f };
	GeneratorParam<int> size{ "size", 512 };
	void generate()
	{
		Var x("x"), y("y");
		Func clamped("clamped");
		clamped = BoundaryConditions::repeat_edge(input);
		Func initialized("initialize");
		initialized(x, y) = select(
			x % 2 == 0 && y % 2 == 0, clamped(x / 2, y / 2),
			0.f
		);
		const float CCiFilter[4] = { -1.f / 16.f, 9.f / 16.f, 9.f / 16.f, -1.f / 16.f };
		Func step1("step1");
		{
			Expr G1; // 45度方向の勾配
			{
				Func norm = calcL1Norm(initialized, { -2, 2 });
				Expr grad = 0.f;
				int mn[3] = { 3, 1, -1 };
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad += norm(x + mn[i], y - mn[j]);
					}
				}
				G1 = grad;
			}
			Expr G2; // 135度方向の勾配
			{
				Func norm = calcL1Norm(initialized, { -2, -2 });
				Expr grad = 0.f;
				int mn[3] = { 3, 1, -1 };
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad += norm(x + mn[i], y + mn[j]);
					}
				}
				G2 = grad;
			}
			Expr W1 = calcWeight(G1);
			Expr W2 = calcWeight(G2);
			Expr pixel1 = initialized(x - 3, y + 3) * CCiFilter[0] + initialized(x - 1, y + 1) * CCiFilter[1] + initialized(x + 1, y - 1) * CCiFilter[2] + initialized(x + 3, y - 3) * CCiFilter[3];
			Expr pixel2 = initialized(x - 3, y - 3) * CCiFilter[0] + initialized(x - 1, y - 1) * CCiFilter[1] + initialized(x + 1, y + 1) * CCiFilter[2] + initialized(x + 3, y + 3) * CCiFilter[3];
			Expr pxSmooth = (W1 * pixel1 + W2 * pixel2) / (W1 + W2);
			Func oddOdd("oddOdd");
			oddOdd(x, y) = select(
				(1.f + G1) / (1.f + G2) > threshold, pixel2,
				(1.f + G2) / (1.f + G1) > threshold, pixel1,
				pxSmooth
			);
			step1(x, y) = select(
				x % 2 == 0 && y % 2 == 0, initialized(x, y),
				x % 2 == 1 && y % 2 == 1, oddOdd(x, y),
				0.f
			);
		}
		Func step2("step2");
		{
			Expr G1; // horizontal
			{
				Func norm = calcL1Norm(step1, { 2, 0 });
				Expr grad = 0.f;
				int m[2] = { 0, 2 };
				int n[2] = { -1, 1 };
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						grad += norm(x - m[i], y + n[j]);
					}
				}
				int n2[3] = { -2, 0, 2 };
				for (int i = 0; i < 3; i++)
				{
					grad += norm(x - 1, y + n2[i]);
				}
				G1 = grad;
			}
			Expr G2; // vertical
			{
				Func norm = calcL1Norm(step1, { 0, 2 });
				Expr grad = 0.f;
				int m[2] = { -1, 1 };
				int n[2] = { 0, 2 };
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						grad += norm(x + m[i], y - n[j]);
					}
				}
				int m2[3] = { -2, 0, 2 };
				for (int i = 0; i < 3; i++)
				{
					grad += norm(x + m2[i], y - 1);
				}
				G2 = grad;
			}
			Expr W1 = calcWeight(G1);
			Expr W2 = calcWeight(G2);
			Expr pixel1 = step1(x - 3, y) * CCiFilter[0] + step1(x - 1, y) * CCiFilter[1] + step1(x + 1, y) * CCiFilter[2] + step1(x + 3, y) * CCiFilter[3];
			Expr pixel2 = step1(x, y - 3) * CCiFilter[0] + step1(x, y - 1) * CCiFilter[1] + step1(x, y + 1) * CCiFilter[2] + step1(x, y + 3) * CCiFilter[3];
			Expr pxSmooth = (W1 * pixel1 + W2 * pixel2) / (W1 + W2);
			Func evenOddAndOddEven("evenOddAndOddEven");
			evenOddAndOddEven(x, y) = select(
				(1.f + G1) / (1.f + G2) > threshold, pixel2,
				(1.f + G2) / (1.f + G1) > threshold, pixel1,
				pxSmooth
			);
			step2(x, y) = select(
				x % 2 == 0 && y % 2 == 0, step1(x, y),
				x % 2 == 1 && y % 2 == 1, step1(x, y),
				x % 2 == 0 && y % 2 == 1, evenOddAndOddEven(x, y),
				x % 2 == 1 && y % 2 == 0, evenOddAndOddEven(x, y),
				0.f
			);
		}
		output(x, y) = step2(x, y);
	}
	void schedule()
	{
		input.set_estimates({ {0, size}, {0, size} });
		output.set_estimates({ {0, size * 2}, {0, size * 2} });
		threshold.set_estimate(1.15f);
	}
};
HALIDE_REGISTER_GENERATOR(DCCI32FC1_Halide_AsPerAlgorithm, DCCI32FC1_Halide_AsPerAlgorithm_Gen)
