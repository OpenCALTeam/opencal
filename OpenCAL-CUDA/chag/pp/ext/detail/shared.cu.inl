/****************************************************************************/
/*!	\brief [chag::pp] shared helper implementation
 */
/* Copyright (c) 2009, Markus Billeter, Ola Olsson and Ulf Assarsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/

//--//////////////////////////////////////////////////////////////////////////
CHAG_PP_ENTER_NAMESPACE()
//--	d :: sm_*			///{{{1///////////////////////////////////////////
namespace detail
{
	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsPlain<T>::value, void) sm_set( 
		volatile T* aSm,
		unsigned aIndex,
		const T& aIn
	)
	{
		aSm[aIndex] = aIn;
	}
	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsPlain<T>::value, T) sm_get( 
		volatile T* aSm,
		unsigned aIndex
	)
	{
		return aSm[aIndex];
	}

	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsPair<T>::value, void) sm_set( 
		volatile T* aSm,
		unsigned aIndex,
		const T& aIn
	)
	{
		aSm[aIndex].x = aIn.x;
		aSm[aIndex].y = aIn.y;
	}
	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsPair<T>::value, T) sm_get( 
		volatile T* aSm,
		unsigned aIndex
	)
	{
		T ret = { aSm[aIndex].x, aSm[aIndex].y };
		return ret;
	}

	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsQuad<T>::value, void) sm_set( 
		volatile T* aSm,
		unsigned aIndex,
		const T& aIn
	)
	{
		aSm[aIndex].x = aIn.x;
		aSm[aIndex].y = aIn.y;
		aSm[aIndex].z = aIn.z;
		aSm[aIndex].w = aIn.w;
	}
	template< typename T > 
	_CHAG_PP_DEV _CHAG_PP_IF(detail::IsQuad<T>::value, T) sm_get( 
		volatile T* aSm,
		unsigned aIndex
	)
	{
		T ret = { aSm[aIndex].x, aSm[aIndex].y, aSm[aIndex].z, aSm[aIndex].w };
		return ret;
	}

}

CHAG_PP_LEAVE_NAMESPACE()
//--///1}}}/////////////// vim:syntax=cuda:foldmethod=marker:ts=4:noexpandtab: 
