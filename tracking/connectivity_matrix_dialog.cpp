#include <QGraphicsTextItem>
#include <QMouseEvent>
#include <QFileDialog>
#include "connectivity_matrix_dialog.h"
#include "ui_connectivity_matrix_dialog.h"
#include "region/regiontablewidget.h"
#include "tract/tracttablewidget.h"
#include "tracking_window.h"
#include "ui_tracking_window.h"
#include "mapping/atlas.hpp"
#include "libs/tracking/fib_data.hpp"
extern std::vector<atlas> atlas_list;
connectivity_matrix_dialog::connectivity_matrix_dialog(tracking_window *parent) :
    QDialog(parent),cur_tracking_window(parent),
    ui(new Ui::connectivity_matrix_dialog)
{
    ui->setupUi(this);
    ui->graphicsView->setScene(&scene);

    std::vector<std::string> index_list;
    cur_tracking_window->handle->get_index_list(index_list);
    for (unsigned int index = 0; index < index_list.size(); ++index)
        ui->matrix_value->addItem(index_list[index].c_str());

    // atlas
    ui->region_list->addItem("ROIs");
    for(int index = 0;index < atlas_list.size();++index)
        ui->region_list->addItem(atlas_list[index].name.c_str());
    if(cur_tracking_window->regionWidget->regions.size() > 1)
        ui->region_list->setCurrentIndex(0);
    else
        if(!atlas_list.empty())
        ui->region_list->setCurrentIndex(1);
    on_recalculate_clicked();

}

connectivity_matrix_dialog::~connectivity_matrix_dialog()
{
    delete ui;
}

bool connectivity_matrix_dialog::is_graphic_view(QObject *object) const
{
    return object == ui->graphicsView;
}
void connectivity_matrix_dialog::mouse_move(QMouseEvent *mouseEvent)
{
    QPointF point = ui->graphicsView->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    int x = std::floor(((float)point.x()) / ui->zoom->value() - 0.5);
    int y = std::floor(((float)point.y()) / ui->zoom->value() - 0.5);
    if(x >= 0 && y >= 0 && x < data.region_name.size() && y < data.region_name.size())
    {
        data.save_to_image(cm);
        // line x
        for(unsigned int x_pos = 0,pos = y*data.matrix_value.height();x_pos < data.matrix_value.width();++x_pos,++pos)
        {
            cm[pos][2] = (cm[pos][0] >> 1);
            cm[pos][2] += 125;
        }
        // line y
        for(unsigned int y_pos = 0,pos = x;y_pos < data.matrix_value.height();++y_pos,pos += data.matrix_value.width())
        {
            cm[pos][2] = (cm[pos][0] >> 1);
            cm[pos][2] += 125;
        }
        on_zoom_valueChanged(0);
        QGraphicsTextItem *x_text = scene.addText(data.region_name[x].c_str());
        QGraphicsTextItem *y_text = scene.addText(data.region_name[y].c_str());
        x_text->moveBy(point.x()-x_text->boundingRect().width()/2,-x_text->boundingRect().height());
        y_text->setRotation(270);
        y_text->moveBy(-y_text->boundingRect().height(),point.y()+y_text->boundingRect().width()/2);
        unsigned int index = x+y*data.matrix_value.width();
        if(index < data.matrix_value.size())
        {
            QGraphicsTextItem *value_text =
                    scene.addText(QString("%1:%2").
                                  arg(ui->matrix_value->currentText()).
                                  arg(data.matrix_value[index]));
            value_text->moveBy(0,-x_text->boundingRect().height()-value_text->boundingRect().height());
        }
    }

}


void connectivity_matrix_dialog::on_recalculate_clicked()
{
    if(cur_tracking_window->tractWidget->tract_models.size() == 0)
        return;
    image::geometry<3> geo = cur_tracking_window->slice.geometry;

    if(ui->region_list->currentIndex() == 0)
        {
            ConnectivityMatrix::region_table_type region_table;
            for(unsigned int index = 0;index < cur_tracking_window->regionWidget->regions.size();++index)
            {
                const std::vector<image::vector<3,short> >& cur_region =
                        cur_tracking_window->regionWidget->regions[index].get();
                image::vector<3,float> pos = std::accumulate(cur_region.begin(),cur_region.end(),image::vector<3,float>(0,0,0));
                pos /= cur_region.size();
                region_table.insert(std::make_pair((float)(pos[0] > (geo[0] >> 1) ? pos[1]-geo[1]:geo[1]-pos[1]),
                                                   std::make_pair(cur_region,std::string(cur_tracking_window->regionWidget->item(index,0)->text().toLocal8Bit().begin()))));
            }
            data.set_regions(region_table);
        }
    else
    {
        if(!cur_tracking_window->can_convert())
            return;
        image::basic_image<image::vector<3,float>,3 > mni_position(geo);
        const FiberDirection& fib = cur_tracking_window->handle->fib;
        for (image::pixel_index<3>index; index.is_valid(geo);index.next(geo))
            if(fib.getFA(index.index(),0) > 0)
            {
                image::vector<3,float> mni((const unsigned int*)index.begin());
                cur_tracking_window->subject2mni(mni);
                mni_position[index.index()] = mni;
            }
        data.set_atlas(atlas_list[ui->region_list->currentIndex()-1],mni_position);
    }
    data.calculate(*(cur_tracking_window->tractWidget->tract_models[cur_tracking_window->tractWidget->currentRow()]),
                   ui->matrix_value->currentText().toLocal8Bit().begin(),
                   ui->end_only->currentIndex());
    data.save_to_image(cm);
    on_zoom_valueChanged(0);
}


void connectivity_matrix_dialog::on_zoom_valueChanged(double arg1)
{
    if(cm.empty())
        return;
    QImage qimage((unsigned char*)&*cm.begin(),cm.width(),cm.height(),QImage::Format_RGB32);
    view_image = qimage.scaled(cm.width()*ui->zoom->value(),cm.height()*ui->zoom->value());
    scene.setSceneRect(0, 0, view_image.width(),view_image.height());
    scene.clear();
    scene.setItemIndexMethod(QGraphicsScene::NoIndex);
    scene.addRect(0, 0, view_image.width(),view_image.height(),QPen(),view_image);
}

void connectivity_matrix_dialog::on_save_as_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                0,
                "Save as",
                cur_tracking_window->absolute_path + "/" +
                cur_tracking_window->tractWidget->item(
                    cur_tracking_window->tractWidget->currentRow(),0)->text() + "_" +
                ui->region_list->currentText() + ".mat",
                "MAT File (*.mat);;Image Files (*.png *.tif *.bmp)");
    if(filename.isEmpty())
        return;
    if(QFileInfo(filename).suffix().toLower() == "mat")
    {
        data.save_to_file(filename.toLocal8Bit().begin());
    }
    else
    {
        data.save_to_image(cm);
        QImage qimage((unsigned char*)&*cm.begin(),cm.width(),cm.height(),QImage::Format_RGB32);
        qimage.save(filename);
    }
}


