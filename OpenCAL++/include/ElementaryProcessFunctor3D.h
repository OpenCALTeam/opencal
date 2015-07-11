// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

#ifndef ELEMENTARYPROCESS3D_H_
#define ELEMENTARYPROCESS3D_H_



class ElementaryProcessFunctor3D {

public:
	ElementaryProcessFunctor3D();
	virtual void run(struct CALModel3D* model, int i, int j,int k) =0;

	void operator()(CALModel3D* model, int i, int j, int k);

	virtual ~ElementaryProcessFunctor3D();
};


#endif /* EMENTARYPROCESS2D_H_ */
