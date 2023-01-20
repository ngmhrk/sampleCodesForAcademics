#include "DCCI.hpp"

using namespace std;
using namespace cv;

void DCCI32FC1(const Mat& src_, Mat& dst, const float threshold, int ompNumThreads)
{
	CV_Assert(src_.type() == CV_32FC1);

	Mat src;
	if (&src_ == &dst)
	{
		src_.copyTo(src);
	}
	else
	{
		src = src_;
	}
	int width = src.cols;
	int height = src.rows;

	dst.create(src.size() * 2, CV_32FC1);

	const __m256 mZero = _mm256_setzero_ps(); //0 0 0 0 / 0 0 0 0
	const __m256 signMask = _mm256_set1_ps(-0.0f); // 0x80000000
	const __m256 mThreshold = _mm256_set1_ps(threshold);
	const __m256 mOnef = _mm256_set1_ps(1.f);
	const __m256 cciCoeff_1 = _mm256_set1_ps(-1.f / 16);
	const __m256 cciCoeff_9 = _mm256_set1_ps(9.f / 16);

	const __m256i offsetSrcLo = _mm256_setr_epi32(0, 0, 1, 0, 2, 0, 3, 0);
	const __m256i offsetSrcHi = _mm256_setr_epi32(4, 0, 5, 0, 6, 0, 7, 0);
	const __m256i offsetDstLo = _mm256_setr_epi32(0, 0, 0, 1, 0, 2, 0, 3);
	const __m256i offsetDstHi = _mm256_setr_epi32(0, 4, 0, 5, 0, 6, 0, 7);


	//原画素の配置と（偶数，偶数）の補間
	//端の処理はとりあえず無視
#pragma omp parallel  for firstprivate(mZero) num_threads(ompNumThreads) schedule(guided)
	for (int y = 1; y < height - 2; y++)
	{
		float* srcp0 = src.ptr<float>(y - 1);
		float* srcp1 = src.ptr<float>(y + 0);
		float* srcp2 = src.ptr<float>(y + 1);
		float* srcp3 = src.ptr<float>(y + 2);

		float* dstp0 = dst.ptr<float>(y * 2);
		float* dstp1 = dst.ptr<float>(y * 2 + 1) + 2;

		for (int x = 0; x < width - 7; x += 8)
		{
			/* --- 1 --- */
			__m256 mSrc = _mm256_load_ps(srcp1);

			__m256 mSrcLo = _mm256_permutevar8x32_ps(mSrc, offsetSrcLo);
			__m256 mSrcHi = _mm256_permutevar8x32_ps(mSrc, offsetSrcHi);

			_mm256_store_ps(dstp0, mSrcLo);
			dstp0 += 8;
			_mm256_store_ps(dstp0, mSrcHi);
			dstp0 += 8;

			__m256 pxUpRight = mZero, pxDownRight = mZero, pxSmooth = mZero;
			__m256 gradUpRight = mZero, gradDownRight = mZero;
			__m256 mTmp0 = mZero, mTmp1 = mZero, mTmp2 = mZero;

			//  0   4   8   C
			//  1   5   9   D
			//  2   6   A   E
			//  3   7   B   F
			__m256 mK0 = _mm256_load_ps(srcp0 + 0);
			__m256 mK1 = _mm256_load_ps(srcp1 + 0);
			__m256 mK2 = _mm256_load_ps(srcp2 + 0);
			__m256 mK3 = _mm256_load_ps(srcp3 + 0);

			__m256 mK4 = _mm256_loadu_ps(srcp0 + 1);
			__m256 mK5 = _mm256_loadu_ps(srcp1 + 1);
			__m256 mK6 = _mm256_loadu_ps(srcp2 + 1);
			__m256 mK7 = _mm256_loadu_ps(srcp3 + 1);

			__m256 mK8 = _mm256_loadu_ps(srcp0 + 2);
			__m256 mK9 = _mm256_loadu_ps(srcp1 + 2);
			__m256 mKA = _mm256_loadu_ps(srcp2 + 2);
			__m256 mKB = _mm256_loadu_ps(srcp3 + 2);

			__m256 mKC = _mm256_loadu_ps(srcp0 + 3);
			__m256 mKD = _mm256_loadu_ps(srcp1 + 3);
			__m256 mKE = _mm256_loadu_ps(srcp2 + 3);
			__m256 mKF = _mm256_loadu_ps(srcp3 + 3);

			/* --- 2(a) --- */
			//UpRight G1
			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK1)));
			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mK5)));
			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK9)));

			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK2)));
			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mK6)));
			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKD, mKA)));

			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK3)));
			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mK7)));
			gradUpRight = _mm256_add_ps(gradUpRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKE, mKB)));

			//DownRight G2
			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5)));
			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));
			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKD)));

			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKA)));
			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK9, mKE)));

			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKB)));
			gradDownRight = _mm256_add_ps(gradDownRight, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKF)));

			/* --- 2(b) --- */
			//G1=gradUpRight
			mTmp0 = _mm256_add_ps(mOnef, gradUpRight);
			mTmp1 = _mm256_add_ps(mOnef, gradDownRight);

			//if (1+G1) / (1+G2) > T then 135
			mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
			__m256 maskEdgeDownRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

			//if (1+G2) / (1+G1) > T then 45
			mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
			__m256 maskEdgeUpRight = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

			/* --- 2(c) --- */
			//UpRight方向にエッジがある場合，UpRight方向に補間 p1
			pxUpRight = _mm256_add_ps(mK3, mKC);
			pxUpRight = _mm256_mul_ps(pxUpRight, cciCoeff_1);
			pxUpRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK6, mK9), pxUpRight);


			//DownRight方向にエッジがある場合，DownRight方向に補間 p2
			pxDownRight = _mm256_add_ps(mK0, mKF);
			pxDownRight = _mm256_mul_ps(pxDownRight, cciCoeff_1);
			pxDownRight = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK5, mKA), pxDownRight);

			//weight = 1 / (1+G^5)
			//weight1はgradUpRightを使う
			__m256 weight1 = _mm256_mul_ps(gradUpRight, gradUpRight);
			weight1 = _mm256_mul_ps(weight1, weight1);
			weight1 = _mm256_fmadd_ps(weight1, gradUpRight, mOnef);
			weight1 = _mm256_rcp_ps(weight1);

			//weight2はgradDownRightを使う
			__m256 weight2 = _mm256_mul_ps(gradDownRight, gradDownRight);
			weight2 = _mm256_mul_ps(weight2, weight2);
			weight2 = _mm256_fmadd_ps(weight2, gradDownRight, mOnef);
			weight2 = _mm256_rcp_ps(weight2);

			//p = (w1p1 + w2p2) / (w1 + w2)
			mTmp2 = _mm256_mul_ps(weight1, pxUpRight);
			mTmp2 = _mm256_fmadd_ps(weight2, pxDownRight, mTmp2);
			pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
			pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);
			mTmp0 = _mm256_add_ps(weight1, weight2);

			//0で最初の引数をとる
			__m256 mDst = _mm256_blendv_ps(pxSmooth, pxUpRight, maskEdgeUpRight);
			mDst = _mm256_blendv_ps(mDst, pxDownRight, maskEdgeDownRight);

			__m256 mDstLo = _mm256_permutevar8x32_ps(mDst, offsetDstLo);
			__m256 mDstHi = _mm256_permutevar8x32_ps(mDst, offsetDstHi);

			_mm256_storeu_ps(dstp1, mDstLo);
			dstp1 += 8;
			_mm256_storeu_ps(dstp1, mDstHi);
			dstp1 += 8;

			srcp0 += 8;
			srcp1 += 8;
			srcp2 += 8;
			srcp3 += 8;
		}//x
	}//y

	height = dst.rows;
	width = dst.cols;

	//（偶数，奇数），（奇数，偶数）の補間
#pragma omp parallel  for firstprivate(mZero) num_threads(ompNumThreads) schedule(guided)
	for (int y = 6; y <= height - 8; y += 2)
	{
		//(y,x)
		float* pEvenOdd = dst.ptr<float>(y);//×
		float* pOddEven = dst.ptr<float>(y + 1);//△

		float* dstp0 = (pOddEven - width * 2 - 1);
		float* dstp1 = (pOddEven - width * 1 - 2);
		float* dstp2 = (pOddEven + width * 0 - 1);
		float* dstp3 = (pOddEven + width * 1 - 2);
		float* dstp4 = (pOddEven + width * 2 - 1);

		for (int x = 0; x <= width - 16; x += 8)
		{
			__m256 mOddEven = _mm256_load_ps(pOddEven);// 0 1 0 2 / 0 3 0 4
			__m256 mEvenOdd = _mm256_load_ps(pEvenOdd);// 0 1 0 2 / 0 3 0 4

			//		#:OddEven		@:EvenOdd
			//			C		|		G		
			//		X 0 X 5 X	|	X C X H X	
			//		1 X 6 @ A	|	0 X 5 X I	
			//	  E X 2 # 7 X F	| 1 X 6 @ A X J
			//		3 X 8 X B	|	2 # 7 X F	
			//		X 4 X 9 X	|	X 8 X B X	
			//			D		|		9		
			__m256 mK0 = _mm256_loadu_ps(dstp0);
			__m256 mK1 = _mm256_loadu_ps(dstp1);
			__m256 mK2 = _mm256_loadu_ps(dstp2);
			__m256 mK3 = _mm256_loadu_ps(dstp3);
			__m256 mK4 = _mm256_loadu_ps(dstp4);

			__m256 mK5 = _mm256_loadu_ps(dstp0 + 2);
			__m256 mK6 = _mm256_loadu_ps(dstp1 + 2);
			__m256 mK7 = _mm256_loadu_ps(dstp2 + 2);
			__m256 mK8 = _mm256_loadu_ps(dstp3 + 2);
			__m256 mK9 = _mm256_loadu_ps(dstp4 + 2);

			__m256 mKA = _mm256_loadu_ps(dstp1 + 4);
			__m256 mKB = _mm256_loadu_ps(dstp3 + 4);

			__m256 mKC = _mm256_loadu_ps(pOddEven - 3 * width);
			__m256 mKD = _mm256_loadu_ps(pOddEven + 3 * width);
			__m256 mKE = _mm256_loadu_ps(pOddEven - 3);
			__m256 mKF = _mm256_loadu_ps(pOddEven + 3);

			__m256 mKG = _mm256_loadu_ps(pOddEven + 1 - 4 * width);
			__m256 mKH = _mm256_loadu_ps(pOddEven + 2 - 3 * width);
			__m256 mKI = _mm256_loadu_ps(pOddEven + 3 - 2 * width);
			__m256 mKJ = _mm256_loadu_ps(pOddEven + 4 - 1 * width);

			//horizontal
			__m256 gradHorizontal = mZero;
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK6)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKA)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK3, mK8)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKB)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK4, mK9)));

			//Vertical
			__m256 gradVertical = mZero;
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK1, mK3)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK2)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK4)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK8)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK7)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK7, mK9)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKB)));

			//Horizontal方向にエッジがある場合，Horizontal方向に補間
			__m256 pxHorizontal = mZero;
			pxHorizontal = _mm256_add_ps(mKE, mKF);
			pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
			pxHorizontal = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK2, mK7), pxHorizontal);

			//Vertical方向にエッジがある場合，Vertical方向に補間
			__m256 pxVertical = mZero;
			pxVertical = _mm256_add_ps(mKC, mKD);
			pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
			pxVertical = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK6, mK8), pxVertical);

			//weight = 1 / (1+G^5)
			//weight1はgradHorizontalを使う
			__m256 weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
			weight1 = _mm256_mul_ps(weight1, weight1);
			weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
			weight1 = _mm256_rcp_ps(weight1);

			//weight2はgradVerticalを使う
			__m256 weight2 = _mm256_mul_ps(gradVertical, gradVertical);
			weight2 = _mm256_mul_ps(weight2, weight2);
			weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
			weight2 = _mm256_rcp_ps(weight2);

			//p = (w1p1+w2p2) / (w1+w2)
			__m256 pxSmooth = mZero;
			__m256 mTmp0 = mZero, mTmp1 = mZero, mTmp2 = mZero;
			mTmp2 = _mm256_mul_ps(weight1, pxHorizontal);
			mTmp2 = _mm256_fmadd_ps(weight2, pxVertical, mTmp2);
			pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));

			pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);

			mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
			mTmp1 = _mm256_add_ps(mOnef, gradVertical);

			//if (1+G1) / (1+G2) > T then 135
			//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
			mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
			__m256 maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

			//if (1+G2) / (1+G1) > T then 45
			//cmpの結果を論理演算に使うとバグる
			mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
			__m256 maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);


			//0で最初の引数をとる
			__m256 mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
			mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

			//blendv
			_mm256_store_ps(pOddEven, _mm256_blend_ps(mDst, mOddEven, 0xAA));

			//horizontal
			gradHorizontal = mZero;
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mKH)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK5)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mKI)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mKA)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK2, mK7)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK7, mKF)));
			gradHorizontal = _mm256_add_ps(gradHorizontal, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK8, mKB)));

			//Vertical
			gradVertical = mZero;
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK0, mK2)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKC, mK6)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK6, mK8)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mK5, mK7)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKH, mKA)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKA, mKB)));
			gradVertical = _mm256_add_ps(gradVertical, _mm256_andnot_ps(signMask, _mm256_sub_ps(mKI, mKF)));

			//Horizontal方向にエッジがある場合，Horizontal方向に補間
			pxHorizontal = mZero;
			pxHorizontal = _mm256_add_ps(mK1, mKJ);
			pxHorizontal = _mm256_mul_ps(cciCoeff_1, pxHorizontal);
			pxHorizontal = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK6, mKA), pxHorizontal);

			//Vertical方向にエッジがある場合，Vertical方向に補間
			pxVertical = mZero;
			pxVertical = _mm256_add_ps(mKG, mK9);
			pxVertical = _mm256_mul_ps(cciCoeff_1, pxVertical);
			pxVertical = _mm256_fmadd_ps(cciCoeff_9, _mm256_add_ps(mK5, mK7), pxVertical);

			//weight = 1 / (1+G^5)
			//weight1はgradHorizontalを使う
			weight1 = _mm256_mul_ps(gradHorizontal, gradHorizontal);
			weight1 = _mm256_mul_ps(weight1, weight1);
			weight1 = _mm256_fmadd_ps(weight1, gradHorizontal, mOnef);
			weight1 = _mm256_rcp_ps(weight1);

			//weight2はgradVerticalを使う
			weight2 = _mm256_mul_ps(gradVertical, gradVertical);
			weight2 = _mm256_mul_ps(weight2, weight2);
			weight2 = _mm256_fmadd_ps(weight2, gradVertical, mOnef);
			weight2 = _mm256_rcp_ps(weight2);


			//p = (w1p1 + w2p2) / (w1 + w2)
			pxSmooth = mZero;
			mTmp2 = _mm256_mul_ps(weight1, pxHorizontal);
			mTmp2 = _mm256_fmadd_ps(weight2, pxVertical, mTmp2);
			pxSmooth = _mm256_rcp_ps(_mm256_add_ps(weight1, weight2));
			pxSmooth = _mm256_mul_ps(pxSmooth, mTmp2);

			mTmp0 = _mm256_add_ps(mOnef, gradHorizontal);//G1=gradHorizontal
			mTmp1 = _mm256_add_ps(mOnef, gradVertical);

			//if (1+G1) / (1+G2) > T then 135
			//(1+G1) / (1+G2) <= T で 0 (=false) が入る？
			mTmp2 = _mm256_mul_ps(mTmp0, _mm256_rcp_ps(mTmp1));
			maskEdgeVertical = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

			//if (1+G2) / (1+G1) > T then 45
			//cmpの結果を論理演算に使うとバグる
			mTmp2 = _mm256_mul_ps(mTmp1, _mm256_rcp_ps(mTmp0));
			maskEdgeHorizontal = _mm256_cmp_ps(mTmp2, mThreshold, _CMP_GT_OS);

			//0で最初の引数をとる
			mDst = _mm256_blendv_ps(pxSmooth, pxHorizontal, maskEdgeHorizontal);
			mDst = _mm256_blendv_ps(mDst, pxVertical, maskEdgeVertical);

			mDst = _mm256_moveldup_ps(mDst);
			_mm256_store_ps(pEvenOdd, _mm256_blend_ps(mEvenOdd, mDst, 0xAA));

			pOddEven += 8;
			pEvenOdd += 8;
			dstp0 += 8;
			dstp1 += 8;
			dstp2 += 8;
			dstp3 += 8;
			dstp4 += 8;

		}//x
	}//y
}